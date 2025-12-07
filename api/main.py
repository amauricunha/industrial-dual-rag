"""FastAPI backend that implements the full Dual-Context RAG workflow.

Highlights for the final deliverable:
        * Problem/use case: responde dúvidas de operação para um torno CNC.
        * Model usage: executa chunking + embeddings + consulta em diferentes
            bancos vetoriais antes de chamar um LLM generativo (Groq, Gemini, Ollama).
        * Experimentos: mede accuracy/BLEU/ROUGE, tokens e latência; gera relatórios
            via botão na UI e endpoints dedicados.
        * Originalidade: heurísticas para seleção de sensores, reindexação sem
            reupload e alternância rápida entre backends vetoriais.
"""

import os
import json
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import pypdf
import pandas as pd
import plotly.express as px
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as FAISSVectorStore

try:
    from langchain_community.vectorstores import Weaviate as WeaviateVectorStore
    import weaviate
except ImportError:  # pragma: no cover - optional dependency
    WeaviateVectorStore = None
    weaviate = None

try:
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
    import pinecone
    from pinecone import ServerlessSpec
except ImportError:  # pragma: no cover - optional dependency
    PineconeVectorStore = None
    pinecone = None
    ServerlessSpec = None

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

load_dotenv()

logging.basicConfig(level=os.getenv("API_LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("industrial-dual-rag-api")

app = FastAPI(title="Industrial Dual-RAG API")

DEFAULT_BASE_SYSTEM = (
    "Você é um Engenheiro Sênior de Diagnóstico Industrial especializado em máquinas de manufatura, com foco em tornos "
    "mecânicos. Analise condições de operação e identifique falhas com base na telemetria e no conteúdo técnico fornecido. "
    "Seja objetivo, técnico e utilize terminologia industrial correta. Quando possível, fundamente suas conclusões utilizando "
    "trechos do contexto."
)

# --- Configurações ---
# DATA_DIR está ligado ao volume ./data/api no docker-compose, preservando uploads,
# índices FAISS e relatórios entre execuções para facilitar os experimentos.
DATA_DIR = "/app/data"
DB_DIR = os.getenv("CHROMA_DB_PATH", os.path.join(DATA_DIR, "chromadb"))
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
KB_Record_File = os.path.join(DATA_DIR, "knowledge_base.json")
EXPERIMENT_LOG = os.path.join(DATA_DIR, "experiment_logs.csv")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", os.path.join(DATA_DIR, "faiss_index"))
SUMMARY_OUTPUT_DIR = os.getenv("SUMMARY_OUTPUT_DIR", os.path.join(DATA_DIR, "summaries"))
SUMMARY_MAX_RECENT = int(os.getenv("SUMMARY_MAX_RECENT", "50"))

DEFAULT_VECTOR_BACKEND = os.getenv("VECTOR_BACKEND_DEFAULT", "chroma").lower()
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_DEFAULT", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_DEFAULT", "200"))
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_DEFAULT", "all-MiniLM-L6-v2")
SUPPORTED_VECTOR_BACKENDS = {"chroma", "faiss", "weaviate", "pinecone"}
DEFAULT_TELEMETRY_KEYS = ["status", "temperature", "vibration", "current"]

_embedding_cache: Dict[str, HuggingFaceEmbeddings] = {}

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Cliente Chroma local garante um fallback rápido mesmo quando backends externos
# não estão disponíveis; ele também suporta o baseline de comparação.
chroma_client = chromadb.PersistentClient(path=DB_DIR)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="industrial_manuals", embedding_function=embedding_func)


def get_embedding_function(model_name: str) -> HuggingFaceEmbeddings:
    """Fornece o embedder solicitado reutilizando cache para evitar reloads caros."""
    if model_name not in _embedding_cache:
        _embedding_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    return _embedding_cache[model_name]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Divide documentos longos em janelas controladas (chunking obrigatório para RAG)."""
    if chunk_size <= 0:
        raise HTTPException(400, "chunk_size deve ser maior que zero.")
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def normalize_telemetry_keys(selected: Optional[List[str]]) -> List[str]:
    """Limita sinais de telemetria às chaves suportadas mantendo ordem determinística."""
    if not selected:
        return DEFAULT_TELEMETRY_KEYS.copy()
    normalized = []
    for key in selected:
        key_lower = (key or "").lower()
        if key_lower in DEFAULT_TELEMETRY_KEYS and key_lower not in normalized:
            normalized.append(key_lower)
    if not normalized:
        return DEFAULT_TELEMETRY_KEYS.copy()
    return normalized


def extract_text_from_pdf(file_path: str) -> str:
    """Lê um PDF e retorna o texto concatenado das páginas."""
    try:
        reader = pypdf.PdfReader(file_path)
    except Exception as exc:
        raise HTTPException(500, f"Erro ao abrir PDF '{os.path.basename(file_path)}': {exc}") from exc

    text = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise HTTPException(500, f"Falha ao extrair texto do PDF '{os.path.basename(file_path)}': {exc}") from exc
        text.append(page_text + "\n")

    joined = "".join(text).strip()
    if not joined:
        raise HTTPException(400, f"PDF '{os.path.basename(file_path)}' sem conteúdo legível.")
    return joined


def generate_experiment_summary(output_dir: Optional[str] = None) -> dict:
    """Agrega métricas e gera gráficos citáveis, atendendo ao critério experimental."""
    csv_path = Path(EXPERIMENT_LOG)
    if not csv_path.exists():
        raise HTTPException(404, "Arquivo experiment_logs.csv não encontrado. Execute experimentos antes de consolidar.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise HTTPException(400, "Nenhum registro disponível para gerar resumo.")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = [
        col
        for col in ["accuracy", "bleu", "rouge_l", "latency_ms", "prompt_tokens", "response_tokens"]
        if col in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = [col for col in ["scenario", "mode_used"] if col in df.columns]
    summary = pd.DataFrame()
    if group_cols:
        aggregations = {col: "mean" for col in numeric_cols}
        count_column = "question" if "question" in df.columns else ("response" if "response" in df.columns else None)
        if count_column:
            aggregations[count_column] = "count"
        summary = (
            df.groupby(group_cols)
            .agg(aggregations)
            .rename(columns={"question": "samples", "response": "samples"})
            .reset_index()
        )
        if "scenario" in summary.columns:
            summary["scenario"] = summary["scenario"].astype(str)

    output_dir_path = Path(output_dir or SUMMARY_OUTPUT_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir_path / "summary_metrics.csv"
    if not summary.empty:
        summary.to_csv(summary_csv_path, index=False)
    else:
        summary_csv_path.write_text("", encoding="utf-8")

    recent_count = min(SUMMARY_MAX_RECENT, len(df))
    recent_csv_path = output_dir_path / "recent_samples.csv"
    df.tail(recent_count).to_csv(recent_csv_path, index=False)

    charts = {}
    if not summary.empty and {"accuracy", "mode_used"}.issubset(summary.columns):
        fig_acc = px.bar(
            summary,
            x="mode_used",
            y="accuracy",
            color="scenario" if "scenario" in summary.columns else None,
            barmode="group",
            title="Accuracy médio por cenário",
        )
        fig_acc.update_xaxes(title_text="Modo")
        fig_acc.update_yaxes(title_text="Accuracy médio")
        accuracy_path = output_dir_path / "accuracy_by_mode.html"
        fig_acc.write_html(accuracy_path)
        charts["accuracy"] = str(accuracy_path)

    if "latency_ms" in df.columns:
        latency_x = "mode_used" if "mode_used" in df.columns else None
        latency_color = "scenario" if "scenario" in df.columns else None
        fig_latency = px.box(
            df,
            x=latency_x,
            y="latency_ms",
            color=latency_color,
            points="all",
            title="Distribuição de latência",
        )
        fig_latency.update_yaxes(title_text="Latência (ms)")
        latency_path = output_dir_path / "latency_distribution.html"
        fig_latency.write_html(latency_path)
        charts["latency"] = str(latency_path)

    metadata_path = output_dir_path / "summary_metadata.json"
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": int(len(df)),
        "summary_rows": int(len(summary)),
        "recent_window": recent_count,
        "output_dir": str(output_dir_path),
        "artifacts": {
            "summary_csv": str(summary_csv_path),
            "recent_samples_csv": str(recent_csv_path),
            "charts": charts,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)

    return metadata


def ensure_backend_supported(backend: str):
    """Valida o backend informado antes de executar operações de RAG."""
    if backend not in SUPPORTED_VECTOR_BACKENDS:
        raise HTTPException(400, f"Backend vetorial '{backend}' não é suportado.")


def upsert_chunks_to_backend(
    backend: str,
    chunks: List[str],
    metadatas: List[dict],
    embedding_model: str,
):
    """Despacha chunks para o backend vetorial escolhido mantendo paridade entre eles."""
    # Mantemos implementações separadas para demonstrar integração com diferentes
    # bases vetoriais exigidas no enunciado (Chroma, FAISS, Weaviate, Pinecone).
    backend = backend.lower()
    ensure_backend_supported(backend)
    if backend == "chroma":
        ids = [f"{meta.get('source', 'manual')}:{idx}" for idx, meta in enumerate(metadatas)]
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        return

    if backend == "faiss":
        embedding = get_embedding_function(embedding_model)
        if os.path.isdir(FAISS_INDEX_DIR):
            store = FAISSVectorStore.load_local(
                FAISS_INDEX_DIR,
                embeddings=embedding,
                allow_dangerous_deserialization=True,
            )
            store.add_texts(chunks, metadatas=metadatas)
        else:
            store = FAISSVectorStore.from_texts(chunks, embedding=embedding, metadatas=metadatas)
        store.save_local(FAISS_INDEX_DIR)
        return

    if backend == "weaviate":
        if not WeaviateVectorStore or not weaviate:
            raise HTTPException(500, "Dependências do Weaviate não instaladas no backend.")
        url = os.getenv("WEAVIATE_URL")
        if not url:
            raise HTTPException(400, "Configure WEAVIATE_URL para usar este backend.")
        api_key = os.getenv("WEAVIATE_API_KEY")
        auth = weaviate.AuthApiKey(api_key) if api_key else None
        client = weaviate.Client(url=url, auth_client_secret=auth)
        index_name = os.getenv("WEAVIATE_CLASS", "IndustrialManual")
        embedding = get_embedding_function(embedding_model)
        store = WeaviateVectorStore(client=client, index_name=index_name, text_key="text", embedding=embedding)
        store.add_texts(chunks, metadatas=metadatas)
        return

    if backend == "pinecone":
        if not PineconeVectorStore or not pinecone:
            raise HTTPException(500, "Dependências do Pinecone não instaladas no backend.")
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        index_name = os.getenv("PINECONE_INDEX", "industrial-dual-rag")
        if not api_key or not environment:
            raise HTTPException(400, "Configure PINECONE_API_KEY e PINECONE_ENVIRONMENT para usar este backend.")
        pc = pinecone.Pinecone(api_key=api_key)
        existing = {item["name"] for item in pc.list_indexes()}
        if index_name not in existing:
            dimension = int(os.getenv("PINECONE_DIMENSION", "384"))
            cloud = os.getenv("PINECONE_CLOUD", "aws")
            region = os.getenv("PINECONE_REGION", "us-east-1")
            if not ServerlessSpec:
                raise HTTPException(500, "pinecone-serverless não disponível.")
            pc.create_index(
                index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        index = pc.Index(index_name)
        embedding = get_embedding_function(embedding_model)
        namespace = os.getenv("PINECONE_NAMESPACE", "default")
        store = PineconeVectorStore(index=index, embedding=embedding, text_key="text", namespace=namespace)
        store.add_texts(chunks, metadatas=metadatas)
        return


def query_backend(
    backend: str,
    question: str,
    embedding_model: str,
    top_k: int = 3,
) -> tuple[List[str], List[dict]]:
    """Executa busca semântica no backend selecionado, retornando textos e metadados."""
    # Consulta simétrica a todos os backends para permitir comparar baseline vs RAG
    # sem alterar o restante da pipeline.
    backend = backend.lower()
    ensure_backend_supported(backend)

    if backend == "chroma":
        results = collection.query(query_texts=[question], n_results=top_k)
        documents = results.get("documents") if results else None
        if documents and documents[0]:
            docs = documents[0]
            metas = (results.get("metadatas") or [[]])[0]
            return docs, metas
        return [], []

    if backend == "faiss":
        if not os.path.isdir(FAISS_INDEX_DIR):
            return [], []
        embedding = get_embedding_function(embedding_model)
        store = FAISSVectorStore.load_local(
            FAISS_INDEX_DIR,
            embeddings=embedding,
            allow_dangerous_deserialization=True,
        )
        docs = store.similarity_search(question, k=top_k)
        return [doc.page_content for doc in docs], [doc.metadata for doc in docs]

    if backend == "weaviate":
        if not WeaviateVectorStore or not weaviate:
            raise HTTPException(500, "Dependências do Weaviate não instaladas no backend.")
        url = os.getenv("WEAVIATE_URL")
        if not url:
            raise HTTPException(400, "Configure WEAVIATE_URL para consultar este backend.")
        api_key = os.getenv("WEAVIATE_API_KEY")
        auth = weaviate.AuthApiKey(api_key) if api_key else None
        client = weaviate.Client(url=url, auth_client_secret=auth)
        index_name = os.getenv("WEAVIATE_CLASS", "IndustrialManual")
        embedding = get_embedding_function(embedding_model)
        store = WeaviateVectorStore(client=client, index_name=index_name, text_key="text", embedding=embedding)
        docs = store.similarity_search(question, k=top_k)
        return [doc.page_content for doc in docs], [doc.metadata for doc in docs]

    if backend == "pinecone":
        if not PineconeVectorStore or not pinecone:
            raise HTTPException(500, "Dependências do Pinecone não instaladas no backend.")
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        index_name = os.getenv("PINECONE_INDEX", "industrial-dual-rag")
        if not api_key or not environment:
            raise HTTPException(400, "Configure PINECONE_API_KEY e PINECONE_ENVIRONMENT para consultar este backend.")
        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        embedding = get_embedding_function(embedding_model)
        namespace = os.getenv("PINECONE_NAMESPACE", "default")
        store = PineconeVectorStore(index=index, embedding=embedding, text_key="text", namespace=namespace)
        docs = store.similarity_search(question, k=top_k)
        return [doc.page_content for doc in docs], [doc.metadata for doc in docs]

    return [], []


def estimate_tokens(text: str) -> int:
    """Estima tokens para monitorar custo/latência mesmo sem contador oficial."""
    content = text or ""
    if not content:
        return 0
    if tiktoken:
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
            return len(encoder.encode(content))
        except Exception:  # pragma: no cover
            pass
    return max(1, len(content.split()))


def safe_float(value, default: float = 0.0) -> float:
    """Converte leituras de telemetria para float garantindo fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_telemetry_section(telemetry: Optional[dict], allowed_keys: Optional[List[str]] = None):
    """Monta o contexto dinâmico respeitando a heurística de sensores selecionados."""
    if not telemetry:
        return "", {}

    normalized = {
        "status": telemetry.get("status", "N/A"),
        "temperature": safe_float(telemetry.get("temperature", 0)),
        "vibration": safe_float(telemetry.get("vibration", 0)),
        "current": safe_float(telemetry.get("current", 0)),
    }

    keys_to_use = normalize_telemetry_keys(allowed_keys)

    telemetry_lines = [
        "=== TELEMETRIA EM TEMPO REAL (CONTEXTO DINÂMICO) ===",
    ]

    filtered_snapshot = {}

    if "status" in keys_to_use:
        telemetry_lines.append(f"- Status Máquina: {normalized['status']}")
        filtered_snapshot["status"] = normalized["status"]
    if "temperature" in keys_to_use:
        telemetry_lines.append(f"- Temperatura: {normalized['temperature']:.1f} °C")
        filtered_snapshot["temperature"] = normalized["temperature"]
    if "vibration" in keys_to_use:
        telemetry_lines.append(f"- Vibração RMS: {normalized['vibration']:.2f} mm/s")
        filtered_snapshot["vibration"] = normalized["vibration"]
    if "current" in keys_to_use:
        telemetry_lines.append(f"- Corrente Motor: {normalized['current']:.1f} A")
        filtered_snapshot["current"] = normalized["current"]

    if "temperature" in keys_to_use and normalized["temperature"] > 90:
        telemetry_lines.append("ALERTA: Temperatura acima do limite crítico.")
    if "vibration" in keys_to_use and normalized["vibration"] > 10:
        telemetry_lines.append("ALERTA: Vibração excessiva detectada.")

    return "\n".join(telemetry_lines) + "\n", filtered_snapshot


def normalize_ollama_base(base_url: Optional[str]) -> str:
    """Normaliza a URL base do Ollama para evitar sufixos duplicados em chamadas."""
    base = (base_url or "http://ollama:11434").rstrip("/")
    if base.lower().endswith("/v1"):
        base = base[:-3]
    return base


def ollama_endpoint(path: str, base_url: Optional[str] = None) -> str:
    """Monta um endpoint completo do Ollama preservando validações básicas de URL."""
    base = normalize_ollama_base(base_url or os.getenv("OLLAMA_BASE_URL"))
    if not base:
        raise ValueError("OLLAMA_BASE_URL não configurada")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"

# --- Integração LLM ---
def get_llm_response(
    provider: str,
    model: Optional[str],
    prompt: str,
    api_key: str = None,
    timeout_seconds: Optional[int] = None,
):
    """Orquestra chamadas Groq/Gemini/Ollama encapsulando diferenças de API."""
    try:
        if provider == "groq":
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                return "Erro: API Key da Groq não configurada."
            client = Groq(api_key=key)
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model or "llama3-8b-8192",
                temperature=0.3
            )
            return chat_completion.choices[0].message.content

        if provider == "gemini":
            key = api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                return "Erro: API Key do Google não configurada."
            genai.configure(api_key=key)
            model_instance = genai.GenerativeModel(model or "gemini-1.5-flash")
            response = model_instance.generate_content(prompt)
            return response.text

        if provider == "local":
            try:
                chat_url = ollama_endpoint("/api/chat")
            except ValueError:
                return "Erro: OLLAMA_BASE_URL não configurada."
            timeout_value = timeout_seconds or int(os.getenv("OLLAMA_CHAT_TIMEOUT", "180"))
            payload = {
                "model": model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama3"),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2},
            }
            try:
                response = requests.post(chat_url, json=payload, timeout=timeout_value)
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                detail = ""
                if exc.response is not None:
                    try:
                        error_json = exc.response.json()
                        detail = error_json.get("error") or error_json.get("message", "")
                    except ValueError:
                        detail = exc.response.text
                return f"Erro na chamada do LLM local: {detail or exc}"

            body = response.json()
            if "message" in body:
                return body["message"].get("content", "")
            choices = body.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return "Erro: Nenhuma resposta retornada pelo LLM local."

        return "Provedor de LLM inválido."
    except Exception as e:
        return f"Erro na chamada do LLM ({provider}): {str(e)}"

# --- Modelos ---
class ChatRequest(BaseModel):
    question: str
    scenario: int 
    telemetry: Optional[dict] = None
    llm_provider: str = "groq"
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    debug: bool = False
    base_system: Optional[str] = None
    instructions: Optional[List[str]] = None
    response_format: Optional[Dict[str, Any]] = None
    llm_timeout: Optional[int] = None
    vector_backend: Optional[str] = None
    embedding_model: Optional[str] = None
    telemetry_signals: Optional[List[str]] = None


class ReindexRequest(BaseModel):
    vector_backend: Optional[str] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    files: Optional[List[str]] = None


class SummaryRequest(BaseModel):
    output_dir: Optional[str] = None


class ExperimentLogEntry(BaseModel):
    question: str
    scenario: int
    llm_provider: str
    llm_model: Optional[str] = None
    response: str
    context_found: bool
    telemetry: Optional[dict] = None
    mode_used: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: Optional[str] = None
    reference_answer: Optional[str] = None
    accuracy: Optional[float] = None
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None
    vector_backend: Optional[str] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None

# --- Endpoints ---

@app.get("/files")
def list_files():
    """Lista PDFs já carregados (elegíveis para reindexação/experimentos)."""
    if os.path.exists(KB_Record_File):
        with open(KB_Record_File, "r") as f: return json.load(f)
    return []

@app.post("/upload")
async def upload_manual(
    file: UploadFile = File(...),
    vector_backend: Optional[str] = Form(None),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
):
    """Recebe um manual PDF, executa chunking/embedding e salva no backend escolhido."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Apenas PDFs.")

    backend_choice = (vector_backend or DEFAULT_VECTOR_BACKEND).lower()
    ensure_backend_supported(backend_choice)

    if chunk_size <= 0:
        raise HTTPException(400, "chunk_size deve ser maior que zero.")
    if chunk_overlap < 0:
        raise HTTPException(400, "chunk_overlap não pode ser negativo.")
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Processamento simples de PDF
    text = extract_text_from_pdf(file_path)
    
    # Chunking/Indexação: parte central do requisito RAG (transforma o PDF em
    # pedaços com metadados prontos para qualquer backend vetorial escolhido).
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise HTTPException(400, "Documento vazio após processamento.")
    metadatas = [
        {
            "source": file.filename,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "backend": backend_choice,
            "embedding_model": embedding_model,
        }
        for _ in chunks
    ]

    try:
        upsert_chunks_to_backend(backend_choice, chunks, metadatas, embedding_model)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Falha ao indexar documento no backend %s", backend_choice)
        raise HTTPException(500, f"Erro ao indexar documento: {exc}") from exc
    
    # Atualiza lista
    current = list_files()
    if file.filename not in current:
        current.append(file.filename)
        with open(KB_Record_File, "w") as f: json.dump(current, f)
            
    return {
        "status": "indexed",
        "chunks": len(chunks),
        "backend": backend_choice,
        "embedding_model": embedding_model,
    }


@app.post("/reindex")
def reindex_manuals(req: ReindexRequest):
    """Reprocessa PDFs já enviados com novas configs de chunking/backend."""
    # Permite repetir experimentos rapidamente quando o pesquisador troca de
    # backend vetorial ou tune de chunking sem reenviar os PDFs.
    backend_choice = (req.vector_backend or DEFAULT_VECTOR_BACKEND).lower()
    ensure_backend_supported(backend_choice)

    chunk_size = req.chunk_size or DEFAULT_CHUNK_SIZE
    chunk_overlap = req.chunk_overlap or DEFAULT_CHUNK_OVERLAP
    embedding_model = req.embedding_model or DEFAULT_EMBEDDING_MODEL

    if chunk_size <= 0:
        raise HTTPException(400, "chunk_size deve ser maior que zero.")
    if chunk_overlap < 0:
        raise HTTPException(400, "chunk_overlap não pode ser negativo.")
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1

    filenames = req.files or list_files()
    if not filenames:
        raise HTTPException(404, "Nenhum manual disponível para reprocessamento.")

    processed = []
    skipped = []
    total_chunks = 0

    for filename in filenames:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.isfile(file_path):
            skipped.append({"file": filename, "reason": "arquivo não encontrado"})
            continue

        try:
            text = extract_text_from_pdf(file_path)
            chunks = chunk_text(text, chunk_size, chunk_overlap)
        except HTTPException as exc:
            skipped.append({"file": filename, "reason": str(exc.detail)})
            continue

        if not chunks:
            skipped.append({"file": filename, "reason": "sem conteúdo após chunking"})
            continue

        metadatas = [
            {
                "source": filename,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "backend": backend_choice,
                "embedding_model": embedding_model,
            }
            for _ in chunks
        ]

        try:
            upsert_chunks_to_backend(backend_choice, chunks, metadatas, embedding_model)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Falha ao reindexar arquivo %s", filename)
            skipped.append({"file": filename, "reason": str(exc)})
            continue

        chunk_count = len(chunks)
        total_chunks += chunk_count
        processed.append({"file": filename, "chunks": chunk_count})

    if not processed:
        detail = "Nenhum manual pôde ser reprocessado."
        if skipped:
            detail += " Consulte os motivos em 'skipped'."
        raise HTTPException(500, detail)

    response = {
        "status": "reindexed",
        "backend": backend_choice,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "total_documents": len(processed),
        "total_chunks": total_chunks,
        "documents": processed,
    }
    if skipped:
        response["skipped"] = skipped
    return response


@app.get("/llm/models")
def list_llm_models(provider: str, api_key: Optional[str] = None):
    """Retorna os modelos disponíveis para cada provedor (Groq, Gemini, Ollama)."""
    provider = provider.lower()

    try:
        if provider == "groq":
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise HTTPException(400, "API Key da Groq não configurada.")
            client = Groq(api_key=key)
            result = client.models.list()
            models = [
                {
                    "id": item.id,
                    "context_window": getattr(item, "context_window", None),
                    "description": getattr(item, "description", ""),
                }
                for item in result.data
            ]
        elif provider == "gemini":
            key = api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise HTTPException(400, "API Key do Google não configurada.")
            genai.configure(api_key=key)
            models = []
            for model_info in genai.list_models():
                methods = getattr(model_info, "supported_generation_methods", [])
                if "generateContent" not in methods:
                    continue
                model_id = model_info.name.split("/")[-1]
                models.append(
                    {
                        "id": model_id,
                        "display_name": getattr(model_info, "display_name", model_id),
                        "input_token_limit": getattr(model_info, "input_token_limit", None),
                    }
                )
        elif provider == "local":
            try:
                tags_url = ollama_endpoint("/api/tags")
            except ValueError:
                raise HTTPException(400, "Defina OLLAMA_BASE_URL para listar modelos locais.")
            try:
                response = requests.get(tags_url, timeout=15)
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                raise HTTPException(502, f"Erro ao consultar Ollama: {exc}") from exc
            payload = response.json()
            data = payload.get("models", [])
            models = []
            for item in data:
                model_id = item.get("name") or item.get("id")
                if not model_id:
                    continue
                details = item.get("details", {})
                models.append(
                    {
                        "id": model_id,
                        "display_name": details.get("display_name") or model_id,
                        "context_window": details.get("context_length") or details.get("ctx"),
                        "description": f"{details.get('parameter_size', '')} {details.get('family', '')}".strip(),
                    }
                )
        else:
            raise HTTPException(400, "Provedor de LLM inválido.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Erro ao listar modelos: {exc}") from exc

    if not models:
        if provider == "local":
            raise HTTPException(
                404,
                "Nenhum modelo local encontrado. Execute 'ollama pull <modelo>' no container ollama e tente novamente."
            )
        raise HTTPException(404, "Nenhum modelo encontrado para este provedor.")

    return {"provider": provider, "models": models}


@app.post("/experiments/log")
def log_experiment(entry: ExperimentLogEntry):
    """Persiste cada diagnóstico com métricas quantitativas para comparação Baseline vs RAG."""
    timestamp = entry.timestamp or datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp": timestamp,
        "question": entry.question,
        "scenario": entry.scenario,
        "mode_used": entry.mode_used or "",
        "llm_provider": entry.llm_provider,
        "llm_model": entry.llm_model or "",
        "vector_backend": entry.vector_backend or "",
        "context_found": entry.context_found,
        "latency_ms": entry.latency_ms or "",
        "reference_answer": (entry.reference_answer or "").strip(),
        "accuracy": entry.accuracy if entry.accuracy is not None else "",
        "bleu": entry.bleu if entry.bleu is not None else "",
        "rouge_l": entry.rouge_l if entry.rouge_l is not None else "",
        "prompt_tokens": entry.prompt_tokens if entry.prompt_tokens is not None else "",
        "response_tokens": entry.response_tokens if entry.response_tokens is not None else "",
        "telemetry": json.dumps(entry.telemetry or {}),
        "response": entry.response.replace("\n", " ").strip(),
    }

    file_exists = os.path.exists(EXPERIMENT_LOG)
    fieldnames = list(row.keys())
    try:
        with open(EXPERIMENT_LOG, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        raise HTTPException(500, f"Falha ao salvar log experimental: {exc}") from exc

    return {"status": "logged", "timestamp": timestamp}


@app.post("/experiments/summarize")
def summarize_experiments(req: SummaryRequest):
    """Executa o pipeline de consolidação (CSV/HTML) para citar no relatório."""
    try:
        summary = generate_experiment_summary(req.output_dir)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Falha ao gerar consolidação de experimentos")
        raise HTTPException(500, f"Erro ao gerar resumo: {exc}") from exc
    return summary

@app.post("/chat")
def run_diagnosis(req: ChatRequest):
    """Executa um diagnóstico completo.

    Entrada:
        - question: pergunta em linguagem natural.
        - scenario: 1=baseline, 2=RAG estático, 3=RAG dual.
        - vector_backend/embedding_model/chunk params (quando relevantes).
        - telemetry/signals: snapshot do simulador + sinais autorizados.

    Saída: texto do LLM + metadados (backend usado, tokens, telemetria aplicada).
    """
    base_system = (req.base_system or DEFAULT_BASE_SYSTEM).strip()
    vector_backend = (req.vector_backend or DEFAULT_VECTOR_BACKEND).lower()
    embedding_model = req.embedding_model or DEFAULT_EMBEDDING_MODEL
    ensure_backend_supported(vector_backend)
    telemetry_keys = normalize_telemetry_keys(req.telemetry_signals)
    
    # Pipeline resumida: (1) opcionalmente injeta chunks de RAG estático,
    # (2) cola sinais de telemetria selecionados, (3) registra metadados para
    # análise, (4) chama o LLM escolhido. Isso evidencia a comparação pedida entre
    # baseline, RAG estático e RAG dual.
    context_part = ""
    telemetry_part = ""
    retrieved_chunks: List[str] = []
    retrieved_metadatas: List[dict] = []
    telemetry_snapshot = {}
    instructions_block = ""
    response_format_block = ""

    if req.instructions:
        instructions_block = "\n=== INSTRUÇÕES OPERACIONAIS ===\n" + "\n".join(req.instructions) + "\n"
    if req.response_format:
        response_format_block = (
            "\n=== FORMATO DE RESPOSTA (JSON) ===\n"
            + json.dumps(req.response_format, indent=2, ensure_ascii=False)
            + "\n"
        )
    
    # RAG Estático (Manuais)
    if req.scenario in [2, 3]:
        try:
            retrieved_chunks, retrieved_metadatas = query_backend(
                vector_backend,
                req.question,
                embedding_model,
                top_k=3,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Falha ao consultar backend %s", vector_backend)
            raise HTTPException(500, f"Erro na busca vetorial: {exc}") from exc

        if retrieved_chunks:
            doc_text = "\n---\n".join(retrieved_chunks)
            if doc_text.strip():
                context_part = f"\n=== BASE DE CONHECIMENTO (MANUAIS TÉCNICOS) ===\n{doc_text}\n"

    # RAG Dinâmico (Telemetria)
    if req.scenario == 3:
        telemetry_part, telemetry_snapshot = build_telemetry_section(req.telemetry, telemetry_keys)

    # Montagem do Prompt Final
    final_prompt = (
        f"{base_system}\n"
        f"{instructions_block}"
        f"{response_format_block}"
        f"{context_part}"
        f"{telemetry_part}"
        f"\n=== SOLICITAÇÃO DO OPERADOR ===\n{req.question}\n\n"
        "Com base APENAS nas informações acima (se fornecidas), gere um relatório de diagnóstico:\n"
        "1. Identificação do Problema (Hipótese)\n"
        "2. Evidência (Dados ou Manual)\n"
        "3. Ação Recomendada"
    )
    
    response_text = get_llm_response(
        req.llm_provider,
        req.llm_model,
        final_prompt,
        req.api_key,
        timeout_seconds=req.llm_timeout,
    )
    prompt_tokens = estimate_tokens(final_prompt)
    response_tokens = estimate_tokens(response_text if isinstance(response_text, str) else str(response_text))
    
    payload = {
        "response": response_text,
        "mode_used": {
            1: "Baseline (Zero-Shot)",
            2: "RAG Estático (Docs)",
            3: "RAG Dual (Docs + Telemetria)"
        }[req.scenario],
        "context_found": bool(retrieved_chunks),
        "vector_backend": vector_backend,
        "token_usage": {
            "prompt": prompt_tokens,
            "response": response_tokens,
        },
        "telemetry_signals": telemetry_keys,
    }

    if req.debug:
        payload["debug"] = {
            "final_prompt": final_prompt,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_metadatas": retrieved_metadatas,
            "telemetry_used": telemetry_snapshot,
            "telemetry_signals": telemetry_keys,
            "llm_call": {
                "provider": req.llm_provider,
                "model": req.llm_model or "auto",
            },
            "prompt_sections": {
                "system": base_system,
                "instructions": req.instructions or [],
                "response_format": req.response_format or {},
                "context": context_part.strip(),
                "telemetry": telemetry_part.strip(),
                "question": req.question,
            },
            "vector_backend": vector_backend,
            "embedding_model": embedding_model,
            "token_usage": {
                "prompt": prompt_tokens,
                "response": response_tokens,
            },
        }

    return payload