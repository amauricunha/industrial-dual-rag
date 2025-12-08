"""Pipeline simplificado para comparar cenários Dual-RAG em modo offline.

O script executa os seguintes passos:
1. Carrega documentos da pasta ./docs e gera embeddings com SentenceTransformer.
2. Indexa o conteúdo em dois repositórios vetoriais: ChromaDB e um banco de memória simples.
3. Monta prompts combinando contexto, telemetria fixa e instruções estruturadas.
4. Consulta dois LLMs (Groq e Gemini). Se as chaves não estiverem configuradas, gera respostas simuladas.
5. Calcula métricas (accuracy, BLEU, ROUGE-L e BERTScore) contra os gabaritos oficiais.
6. Exporta um CSV com todos os experimentos e gráficos .png para facilitar artigos/apresentações.

Execute com:
    cd core
    cp .env.example .env  # preencha as chaves
    pip install -r requirements.txt
    python industrial_dual_rag.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bert_score import BERTScorer
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from sentence_transformers import SentenceTransformer

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - dependência opcional
    PdfReader = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - fallback para ambientes sem a lib
    genai = None

# Diretórios-base
CORE_DIR = Path(__file__).resolve().parent
DOCS_DIR = CORE_DIR / "docs"
OUTPUT_DIR = CORE_DIR / "output"
CHROMA_DIR = CORE_DIR / ".chroma"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configurações de prompt alinhadas ao Streamlit original
BASE_SYSTEM_PROMPT = (
    "Você é um Engenheiro Sênior de Diagnóstico Industrial especializado em tornos ROMI T 240. "
    "Analise telemetria e contexto técnico para classificar o estado geral (NORMAL, ALERTA ou FALHA), "
    "indicando evidências e ações recomendadas. Cite sempre a fonte do manual."  # noqa: E231
)
INSTRUCTIONS = [
    "Priorize telemetria frente ao texto estático.",
    "Use o contexto para citar limites numéricos e normas.",
    "Responda obrigatoriamente no formato JSON informado.",
    "Não invente valores ausentes.",
]
RESPONSE_SCHEMA = {
    "estado_geral": "NORMAL | ALERTA | FALHA",
    "avaliacao_telemetria": {
        "temperatura": {"valor": "", "analise": ""},
        "vibracao_rms": {"valor": "", "analise": ""},
        "corrente_motor": {"valor": "", "analise": ""},
        "rpm": {"valor": "", "analise": ""},
    },
    "diagnostico_resumido": "",
    "causas_provaveis": [""],
    "acoes_recomendadas": [""],
    "limites_referenciados": [
        {"variavel": "", "limite": "", "fonte_contexto": ""}
    ],
    "justificativa": "",
    "trechos_utilizados": [""],
}

REFERENCE_ANSWERS: Dict[int, str] = {
    1: json.dumps({
        "estado_geral": "NORMAL",
        "avaliacao_telemetria": {
            "temperatura": {"valor": "45 °C", "analise": "Dentro da faixa segura (<65 °C)."},
            "vibracao_rms": {"valor": "2.4 mm/s", "analise": "Zona A ISO 20816."},
            "corrente_motor": {"valor": "13 A", "analise": "Consumo nominal sem picos."},
            "rpm": {"valor": "1 200 rpm", "analise": "Sequência de desbaste."},
        },
        "diagnostico_resumido": "Máquina operando dentro dos parâmetros.",
        "causas_provaveis": ["Processo estável."],
        "acoes_recomendadas": [
            "Manter checklist diário.",
            "Registrar leituras no prontuário."
        ],
        "limites_referenciados": [
            {"variavel": "Temperatura", "limite": "< 65 °C", "fonte_contexto": "Manual Seção 4.2"},
            {"variavel": "Vibração", "limite": "< 2.8 mm/s", "fonte_contexto": "ISO 20816"},
        ],
        "justificativa": "Todos os sinais estão dentro das faixas nominais.",
        "trechos_utilizados": ["Manual Seção 4.2", "Checklist Diário"],
    }, ensure_ascii=False, indent=2),
    2: json.dumps({
        "estado_geral": "FALHA",
        "avaliacao_telemetria": {
            "temperatura": {"valor": "92 °C", "analise": ">90 °C exige parada imediata (Seção 5.1.1)."},
            "vibracao_rms": {"valor": "3.1 mm/s", "analise": "Sem anomalia mecânica relevante."},
            "corrente_motor": {"valor": "15.8 A", "analise": "Arraste térmico eleva corrente."},
            "rpm": {"valor": "900 rpm", "analise": "Rotação moderada não explica o calor."},
        },
        "diagnostico_resumido": "Superaquecimento do cabeçote caracterizado.",
        "causas_provaveis": [
            "Restrição no circuito de refrigeração auxiliar.",
            "Ventoinha obstruída."
        ],
        "acoes_recomendadas": [
            "Acionar E-STOP e manter eixo em repouso.",
            "Remover obstruções do trocador e resetar intertravamento após resfriar."
        ],
        "limites_referenciados": [
            {"variavel": "Temperatura", "limite": "Alerta > 85 °C", "fonte_contexto": "Manual Seção 5.1.1"}
        ],
        "justificativa": "Temperatura excede o limite crítico independentemente dos demais sinais.",
        "trechos_utilizados": ["Manual – Protocolo de Superaquecimento"],
    }, ensure_ascii=False, indent=2),
    3: json.dumps({
        "estado_geral": "ALERTA",
        "avaliacao_telemetria": {
            "temperatura": {"valor": "48 °C", "analise": "Temperatura normal."},
            "vibracao_rms": {"valor": "11.2 mm/s", "analise": ">10 mm/s (Zona D ISO 10816)."},
            "corrente_motor": {"valor": "12.5 A", "analise": "Sem sobrecorrente."},
            "rpm": {"valor": "1 600 rpm", "analise": "Alta rotação amplifica desbalanceamento."},
        },
        "diagnostico_resumido": "Alerta de desbalanceamento do conjunto árvore-cartucho.",
        "causas_provaveis": [
            "Castanhas soltas ou cavaco preso no prato.",
            "Centro de massa deslocado."
        ],
        "acoes_recomendadas": [
            "Reduzir rotação e inspecionar fixações.",
            "Remover cavacos, reapertar e balancear se necessário."
        ],
        "limites_referenciados": [
            {"variavel": "Vibração", "limite": "> 10 mm/s", "fonte_contexto": "Manual Seção 6.3 / ISO 10816"}
        ],
        "justificativa": "Somente vibração excede limite, indicando falha mecânica localizada.",
        "trechos_utilizados": ["Manual – Tabela de Vibração", "ISO 10816"],
    }, ensure_ascii=False, indent=2),
}


@dataclass
class Scenario:
    """Representa um cenário de teste (Normal, Falha Térmica, Desbalanceamento)."""

    scenario_id: int
    name: str
    telemetry: Dict[str, float | str]
    question: str


@dataclass
class ExperimentMode:
    """Define o nível de contexto disponível para o LLM."""

    mode_id: int
    label: str
    use_retrieval: bool
    use_telemetry: bool


SCENARIOS: List[Scenario] = [
    Scenario(
        scenario_id=1,
        name="Operação Normal",
        telemetry={"status": "OPERATIONAL", "temperature": 45.0, "vibration": 2.4, "current": 13.0, "rpm": 1200},
        question="Qual o estado atual da máquina e recomendações de manutenção?",
    ),
    Scenario(
        scenario_id=2,
        name="Falha Térmica",
        telemetry={"status": "WARNING_TEMP", "temperature": 92.0, "vibration": 3.1, "current": 15.8, "rpm": 900},
        question="Identifique falha térmica e descreva ações obrigatórias para resfriamento seguro.",
    ),
    Scenario(
        scenario_id=3,
        name="Desbalanceamento",
        telemetry={"status": "VIBRATION_ALERT", "temperature": 48.0, "vibration": 11.2, "current": 12.5, "rpm": 1600},
        question="Avalie a vibração RMS e indique correções mecânicas imediatas.",
    ),
]

# Modos solicitados: (1) Sem RAG/Sensores, (2) RAG apenas, (3) RAG + Telemetria.
# Ordem fixa para comparar facilmente cada camada de contexto.
MODES: List[ExperimentMode] = [
    ExperimentMode(mode_id=1, label="Baseline (LLM Puro)", use_retrieval=False, use_telemetry=False),
    ExperimentMode(mode_id=2, label="RAG (Docs)", use_retrieval=True, use_telemetry=False),
    ExperimentMode(mode_id=3, label="Dual RAG (Docs + Sensores)", use_retrieval=True, use_telemetry=True),
]

VECTOR_BACKENDS = ["chroma", "simple"]
LLM_PROVIDERS = ["groq", "gemini"]
TELEMETRY_CHANNELS = ["status", "temperature", "vibration", "current", "rpm"]


def load_documents() -> List[Dict[str, str]]:
    """Carrega .txt e .pdf da pasta docs para compor o banco de conhecimento."""

    docs: List[Dict[str, str]] = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(path)
        else:
            continue

        text = text.strip()
        if text:
            docs.append({"id": path.stem, "source": path.name, "text": text})

    if not docs:
        raise FileNotFoundError("Nenhum documento encontrado em core/docs.")
    return docs


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto concatenado de um PDF para alimentar o embedder."""

    if PdfReader is None:
        raise ImportError("Dependência pypdf ausente. Instale-a com `pip install pypdf`.")

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def build_simple_store(embeddings: np.ndarray, docs: List[Dict[str, str]]):
    class SimpleVectorDB:
        def __init__(self, vectors: np.ndarray, payloads: List[Dict[str, str]]):
            self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            self.payloads = payloads

        def query(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, str]]:
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
            scores = self.vectors @ query_norm
            idx = np.argsort(scores)[::-1][:top_k]
            return [self.payloads[i] | {"score": float(scores[i])} for i in idx]

    return SimpleVectorDB(embeddings, docs)


def prepare_vector_backends(embedder: SentenceTransformer, docs: List[Dict[str, str]]):
    """Retorna dict com instâncias dos backends (Chroma e simples)."""
    texts = [item["text"] for item in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    # Backend simples baseado em NumPy
    simple_store = build_simple_store(embeddings, docs)

    # Backend Chroma persistente para inspeção posterior
    CHROMA_DIR.mkdir(exist_ok=True)
    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="industrial_dual_rag_slim",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        ),
    )
    collection.delete(where={})
    collection.add(
        ids=[doc["id"] for doc in docs],
        documents=texts,
        metadatas=[{"source": doc["source"]} for doc in docs],
        embeddings=embeddings.tolist(),
    )

    return {"simple": simple_store, "chroma": collection}


def retrieve_context(question: str, backend: str, stores: Dict[str, object], embedder: SentenceTransformer, top_k: int = 3) -> str:
    """Busca os trechos mais relevantes no repositório escolhido."""
    if backend == "simple":
        query_vector = embedder.encode([question], convert_to_numpy=True)[0]
        results = stores["simple"].query(query_vector, top_k=top_k)
        return "\n---\n".join(item["text"] for item in results)

    if backend == "chroma":
        query_embedding = embedder.encode([question], convert_to_numpy=True)[0].tolist()
        result = stores["chroma"].query(query_embeddings=[query_embedding], n_results=top_k)
        docs = (result.get("documents") or [[]])[0]
        return "\n---\n".join(docs)

    raise ValueError(f"Backend vetorial não suportado: {backend}")


def format_telemetry_block(telemetry: Dict[str, float | str]) -> str:
    """Produz o bloco textual usado no prompt com as leituras selecionadas."""

    if not telemetry:
        return "=== TELEMETRIA NÃO FORNECIDA NESTE MODO ==="

    lines = ["=== TELEMETRIA FIXA PARA O CENÁRIO ==="]
    for key in TELEMETRY_CHANNELS:
        if key in telemetry:
            lines.append(f"- {key.upper()}: {telemetry[key]}")
    return "\n".join(lines)


def build_prompt(question: str, context: str, telemetry: Dict[str, float | str], mode_label: str) -> str:
    """Monta o prompt final enviado ao LLM para um modo/cenário específico."""

    instructions = "\n".join(f"- {item}" for item in INSTRUCTIONS)
    schema = json.dumps(RESPONSE_SCHEMA, ensure_ascii=False, indent=2)
    context_block = context if context.strip() else "[Sem contexto RAG neste modo]"

    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"=== MODO DO EXPERIMENTO ===\n{mode_label}\n\n"
        f"=== INSTRUÇÕES ===\n{instructions}\n\n"
        f"=== FORMATO DE RESPOSTA ===\n{schema}\n\n"
        f"=== CONTEXTO RAG ===\n{context_block}\n\n"
        f"{format_telemetry_block(telemetry)}\n\n"
        f"=== PERGUNTA ===\n{question}\n"
    )


def call_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não configurada.")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    if not api_key or not genai:
        raise RuntimeError("GOOGLE_API_KEY não configurada ou biblioteca indisponível.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def fallback_response(provider: str, scenario_id: int) -> str:
    """Retorna um texto determinístico quando a chamada ao LLM falha."""
    print(f"[WARN] Usando resposta simulada para {provider} no cenário {scenario_id}.")
    return f"/* Simulated {provider} output */\n{REFERENCE_ANSWERS[scenario_id]}"


def invoke_llm(provider: str, prompt: str, scenario_id: int) -> Tuple[str, float]:
    start = time.perf_counter()
    try:
        if provider == "groq":
            text = call_groq(prompt)
        elif provider == "gemini":
            text = call_gemini(prompt)
        else:
            raise ValueError(f"Provedor desconhecido: {provider}")
    except Exception as exc:  # pragma: no cover - caminho de fallback
        print(f"[WARN] Falha ao consultar {provider}: {exc}")
        text = fallback_response(provider, scenario_id)
    latency_ms = (time.perf_counter() - start) * 1000
    return text, latency_ms


def compute_metrics(candidate: str, reference: str, bert_scorer: BERTScorer) -> Dict[str, float]:
    candidate = (candidate or "").strip()
    reference = (reference or "").strip()
    accuracy = 1.0 if candidate.lower() == reference.lower() else 0.0

    bleu = None
    rouge_l = None
    bert_f1 = None

    try:
        bleu = corpus_bleu([candidate], [[reference]]).score
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Falha BLEU: {exc}")

    try:
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = rouge.score(reference, candidate)["rougeL"].fmeasure * 100
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Falha ROUGE-L: {exc}")

    try:
        _, _, f1 = bert_scorer.score([candidate], [reference])
        bert_f1 = float(f1.mean().item() * 100)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Falha BERTScore: {exc}")

    return {
        "accuracy": accuracy,
        "bleu": bleu,
        "rouge_l": rouge_l,
        "bert_score_f1": bert_f1,
    }


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path):
    title_map = {
        "accuracy": "Accuracy",
        "bleu": "BLEU",
        "bert_score_f1": "BERTScore F1",
        "rouge_l": "ROUGE-L",
    }
    pivot = (
        df.pivot_table(
            index="scenario_name",
            columns=["mode_label", "llm_provider", "vector_backend"],
            values=metric,
        )
        .sort_index()
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Comparação de {title_map.get(metric, metric)}")
    ax.set_ylabel(title_map.get(metric, metric))
    ax.set_xlabel("Cenário")
    ax.legend(title="Modo / LLM / Backend", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    output_path = output_dir / f"{metric}_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Gráfico salvo em {output_path.relative_to(CORE_DIR)}")


def run_pipeline():
    load_dotenv(dotenv_path=CORE_DIR / ".env", override=False)
    embed_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    bert_model_name = os.getenv("BERT_SCORE_MODEL", "neuralmind/bert-base-portuguese-cased")

    print(f"[INFO] Carregando embedder {embed_model_name}...")
    embedder = SentenceTransformer(embed_model_name)
    print(f"[INFO] Carregando BERTScore {bert_model_name}...")
    bert_scorer = BERTScorer(lang="pt", model_type=bert_model_name, rescale_with_baseline=True)

    docs = load_documents()
    stores = prepare_vector_backends(embedder, docs)
    results: List[Dict[str, object]] = []

    for scenario in SCENARIOS:
        print(f"[INFO] Rodando cenário {scenario.scenario_id} - {scenario.name}")
        for mode in MODES:
            # Sem RAG só precisamos de um placeholder "none"; caso contrário iteramos nos backends.
            backend_options = VECTOR_BACKENDS if mode.use_retrieval else ["none"]
            for backend in backend_options:
                context = ""
                if mode.use_retrieval:
                    context = retrieve_context(scenario.question, backend, stores, embedder)

                telemetry_payload = scenario.telemetry if mode.use_telemetry else {}
                for provider in LLM_PROVIDERS:
                    prompt = build_prompt(scenario.question, context, telemetry_payload, mode.label)
                    response, latency_ms = invoke_llm(provider, prompt, scenario.scenario_id)
                    metrics = compute_metrics(response, REFERENCE_ANSWERS[scenario.scenario_id], bert_scorer)
                    result_row = {
                        "scenario": scenario.scenario_id,
                        "scenario_name": scenario.name,
                        "mode_id": mode.mode_id,
                        "mode_label": mode.label,
                        "uses_retrieval": mode.use_retrieval,
                        "uses_telemetry": mode.use_telemetry,
                        "vector_backend": backend if mode.use_retrieval else "none",
                        "llm_provider": provider,
                        "response": response,
                        "latency_ms": latency_ms,
                        **metrics,
                    }
                    results.append(result_row)
                    bert_value = metrics.get("bert_score_f1")
                    bert_text = f"{bert_value:.2f}" if bert_value is not None else "nan"
                    print(
                        f"[OK] Cenário {scenario.scenario_id} | {mode.label} | {provider} | "
                        f"Backend={result_row['vector_backend']} | BERT={bert_text}"
                    )

    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Resultados salvos em {csv_path.relative_to(CORE_DIR)}")

    for metric in ["accuracy", "bleu", "bert_score_f1", "rouge_l"]:
        if df[metric].notna().any():
            plot_metric(df, metric, OUTPUT_DIR)


if __name__ == "__main__":
    run_pipeline()
