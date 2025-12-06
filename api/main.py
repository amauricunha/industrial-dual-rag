import os
import json
import csv
from datetime import datetime, timezone
from typing import Optional, List

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import pypdf
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Industrial Dual-RAG API")

# --- Configurações ---
DATA_DIR = "/app/data"
DB_DIR = os.getenv("CHROMA_DB_PATH", os.path.join(DATA_DIR, "chromadb"))
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
KB_Record_File = os.path.join(DATA_DIR, "knowledge_base.json")
EXPERIMENT_LOG = os.path.join(DATA_DIR, "experiment_logs.csv")

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_DIR)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="industrial_manuals", embedding_function=embedding_func)


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_telemetry_section(telemetry: Optional[dict]):
    """Return prompt section text plus normalized telemetry snapshot."""
    if not telemetry:
        return "", {}

    normalized = {
        "status": telemetry.get("status", "N/A"),
        "temperature": safe_float(telemetry.get("temperature", 0)),
        "vibration": safe_float(telemetry.get("vibration", 0)),
        "current": safe_float(telemetry.get("current", 0)),
    }

    telemetry_lines = [
        "=== TELEMETRIA EM TEMPO REAL (CONTEXTO DINÂMICO) ===",
        f"- Status Máquina: {normalized['status']}",
        f"- Temperatura: {normalized['temperature']:.1f} °C",
        f"- Vibração RMS: {normalized['vibration']:.2f} mm/s",
        f"- Corrente Motor: {normalized['current']:.1f} A",
    ]

    if normalized["temperature"] > 90:
        telemetry_lines.append("ALERTA: Temperatura acima do limite crítico.")
    if normalized["vibration"] > 10:
        telemetry_lines.append("ALERTA: Vibração excessiva detectada.")

    return "\n".join(telemetry_lines) + "\n", normalized


def normalize_ollama_base(base_url: Optional[str]) -> str:
    base = (base_url or "http://ollama:11434").rstrip("/")
    if base.lower().endswith("/v1"):
        base = base[:-3]
    return base


def ollama_endpoint(path: str, base_url: Optional[str] = None) -> str:
    base = normalize_ollama_base(base_url or os.getenv("OLLAMA_BASE_URL"))
    if not base:
        raise ValueError("OLLAMA_BASE_URL não configurada")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"

# --- Integração LLM ---
def get_llm_response(provider: str, model: Optional[str], prompt: str, api_key: str = None):
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
            payload = {
                "model": model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama3"),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2},
            }
            response = requests.post(chat_url, json=payload, timeout=90)
            response.raise_for_status()
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

# --- Endpoints ---

@app.get("/files")
def list_files():
    if os.path.exists(KB_Record_File):
        with open(KB_Record_File, "r") as f: return json.load(f)
    return []

@app.post("/upload")
async def upload_manual(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Apenas PDFs.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Processamento simples de PDF
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        raise HTTPException(500, f"Erro leitura PDF: {str(e)}")
    
    # Chunking (simples para o projeto)
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename} for _ in chunks]
    
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    
    # Atualiza lista
    current = list_files()
    if file.filename not in current:
        current.append(file.filename)
        with open(KB_Record_File, "w") as f: json.dump(current, f)
            
    return {"status": "indexed", "chunks": len(chunks)}


@app.get("/llm/models")
def list_llm_models(provider: str, api_key: Optional[str] = None):
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
        raise HTTPException(404, "Nenhum modelo encontrado para este provedor.")

    return {"provider": provider, "models": models}


@app.post("/experiments/log")
def log_experiment(entry: ExperimentLogEntry):
    timestamp = entry.timestamp or datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp": timestamp,
        "question": entry.question,
        "scenario": entry.scenario,
        "mode_used": entry.mode_used or "",
        "llm_provider": entry.llm_provider,
        "llm_model": entry.llm_model or "",
        "context_found": entry.context_found,
        "latency_ms": entry.latency_ms or "",
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

@app.post("/chat")
def run_diagnosis(req: ChatRequest):
    # Prompt Engineering Científico/Técnico
    base_system = (
        "Você é um Engenheiro Sênior de Diagnóstico Industrial. "
        "Sua tarefa é analisar falhas em máquinas rotativas. "
        "Seja técnico, direto e cite as fontes se disponíveis."
    )
    
    context_part = ""
    telemetry_part = ""
    retrieved_chunks: List[str] = []
    retrieved_metadatas: List[dict] = []
    telemetry_snapshot = {}
    
    # RAG Estático (Manuais)
    if req.scenario in [2, 3]:
        results = collection.query(query_texts=[req.question], n_results=3)
        documents = results.get('documents') if results else None
        if documents and documents[0]:
            retrieved_chunks = documents[0]
            metadatas = results.get('metadatas') or []
            retrieved_metadatas = metadatas[0] if metadatas else []
            doc_text = "\n---\n".join(retrieved_chunks)
            if doc_text.strip():
                context_part = f"\n=== BASE DE CONHECIMENTO (MANUAIS TÉCNICOS) ===\n{doc_text}\n"

    # RAG Dinâmico (Telemetria)
    if req.scenario == 3:
        telemetry_part, telemetry_snapshot = build_telemetry_section(req.telemetry)

    # Montagem do Prompt Final
    final_prompt = (
        f"{base_system}\n"
        f"{context_part}"
        f"{telemetry_part}"
        f"\n=== SOLICITAÇÃO DO OPERADOR ===\n{req.question}\n\n"
        "Com base APENAS nas informações acima (se fornecidas), gere um relatório de diagnóstico:\n"
        "1. Identificação do Problema (Hipótese)\n"
        "2. Evidência (Dados ou Manual)\n"
        "3. Ação Recomendada"
    )
    
    response_text = get_llm_response(req.llm_provider, req.llm_model, final_prompt, req.api_key)
    
    payload = {
        "response": response_text,
        "mode_used": {
            1: "Baseline (Zero-Shot)",
            2: "RAG Estático (Docs)",
            3: "RAG Dual (Docs + Telemetria)"
        }[req.scenario],
        "context_found": bool(retrieved_chunks)
    }

    if req.debug:
        payload["debug"] = {
            "final_prompt": final_prompt,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_metadatas": retrieved_metadatas,
            "telemetry_used": telemetry_snapshot,
            "llm_call": {
                "provider": req.llm_provider,
                "model": req.llm_model or "auto",
            },
            "prompt_sections": {
                "system": base_system,
                "context": context_part.strip(),
                "telemetry": telemetry_part.strip(),
                "question": req.question,
            },
        }

    return payload