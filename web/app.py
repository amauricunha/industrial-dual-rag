"""Streamlit dashboard for configuring, testing, and logging the Dual-RAG pipeline.

Relevância para a entrega:
        * Expõe parâmetros de chunking/embedding, troca de backend vetorial e
            seleção de sensores, permitindo comparar Baseline vs. RAG Estático vs. Dual.
        * Oferece elementos experimentais: checkbox de logging, captura de gabarito
            e botão de consolidação de métricas.
        * Serve como interface de demonstração (upload → injeção de falhas →
            diagnóstico), cobrindo o requisito de problema realista.
"""

import streamlit as st
import requests
import json
import os
import time
import logging
from queue import Queue, Empty
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from typing import Optional
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
logger = logging.getLogger("web-app")
LOG_MQTT_EVENTS = os.getenv("LOG_MQTT_EVENTS", "true").lower() == "true"

def env_or_default(*keys, default=None):
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


# --- Setup ---
API_URL = os.getenv("API_URL", "http://api:8000")
MQTT_BROKER = env_or_default("MQTT_BROKER_ADDRESS", "MQTT_BROKER", default="test.mosquitto.org")
MQTT_PORT = int(env_or_default("MQTT_BROKER_PORT", default=1883))
MQTT_TOPIC_DATA = env_or_default("MQTT_TOPIC_SENSORS", "MQTT_TOPIC", default="industrial/lathe/sensors")
MQTT_TOPIC_CMD = env_or_default("MQTT_TOPIC_COMMANDS", default="industrial/lathe/commands")
MQTT_USER = os.getenv("MQTT_USERNAME")
MQTT_PASS = os.getenv("MQTT_PASSWORD")
VECTOR_BACKEND_OPTIONS = ["chroma", "faiss", "weaviate", "pinecone"]
DEFAULT_VECTOR_BACKEND = os.getenv("VECTOR_BACKEND_DEFAULT", "chroma").lower()
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_DEFAULT", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_DEFAULT", "200"))
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_DEFAULT", "all-MiniLM-L6-v2")
DEFAULT_BASE_SYSTEM = (
    "Você é um Engenheiro Sênior de Diagnóstico Industrial especializado em máquinas de manufatura, com foco em tornos "
    "mecânicos. Analise condições de operação e identifique falhas com base na telemetria e no conteúdo técnico "
    "fornecido. Seja objetivo, técnico e utilize terminologia industrial correta. Quando possível, fundamente suas "
    "conclusões utilizando trechos do contexto."
)
DEFAULT_INSTRUCTIONS = "\n".join(
    [
        "1. Sempre priorize valores de telemetria frente ao texto do contexto.",
        "2. Use o contexto apenas como referência para limites e recomendações.",
        "3. Classifique o estado geral entre NORMAL, ALERTA ou FALHA.",
        "4. Justifique todas as decisões usando valores numéricos e trechos do contexto.",
        "5. Nunca invente valores ou limites que não estiverem na telemetria ou no contexto.",
        "6. Sempre cite a fonte do contexto entre colchetes quando possível.",
        "7. Responda obrigatoriamente no formato JSON especificado a seguir.",
    ]
)
DEFAULT_RESPONSE_FORMAT = {
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
    "limites_referenciados": [{"variavel": "", "limite": "", "fonte_contexto": ""}],
    "justificativa": "",
    "trechos_utilizados": [""],
}
DEFAULT_RESPONSE_FORMAT_TEXT = json.dumps(DEFAULT_RESPONSE_FORMAT, indent=2, ensure_ascii=False)
DEFAULT_LLM_TIMEOUT = int(os.getenv("OLLAMA_CHAT_TIMEOUT", "180"))
BERT_SCORE_MODEL = os.getenv("BERT_SCORE_MODEL", "neuralmind/bert-base-portuguese-cased")
BERT_SCORE_FALLBACK_MODEL = "xlm-roberta-base"
TELEMETRY_SIGNAL_OPTIONS = [
    ("status", "Status da máquina"),
    ("temperature", "Temperatura (°C)"),
    ("vibration", "Vibração RMS (mm/s)"),
    ("current", "Corrente do motor (A)"),
]
TELEMETRY_SIGNAL_DEFAULTS = [opt[0] for opt in TELEMETRY_SIGNAL_OPTIONS]

st.set_page_config(page_title="Industrial Dual-RAG Lab", layout="wide")

# --- Estado ---
# Mantemos os parâmetros do experimento no session_state para permitir que o
# professor reproduza rapidamente diferentes combinações de prompts/sensores.
if "telemetry" not in st.session_state:
    st.session_state.telemetry = {"temperature": 0, "vibration": 0, "current": 0, "status": "OFFLINE"}
if "diagnosis_history" not in st.session_state:
    st.session_state.diagnosis_history = None
if "mqtt_error" not in st.session_state:
    st.session_state.mqtt_error = None
if "llm_model_choice" not in st.session_state:
    st.session_state.llm_model_choice = ""
if "model_cache_key" not in st.session_state:
    st.session_state.model_cache_key = None
if "model_cache" not in st.session_state:
    st.session_state.model_cache = []
if "last_telemetry_reading" not in st.session_state:
    st.session_state.last_telemetry_reading = None
if "last_telemetry_time" not in st.session_state:
    st.session_state.last_telemetry_time = None
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if "auto_refresh_interval" not in st.session_state:
    st.session_state.auto_refresh_interval = 2.0
if "base_system_text" not in st.session_state:
    st.session_state.base_system_text = DEFAULT_BASE_SYSTEM
if "instructions_text" not in st.session_state:
    st.session_state.instructions_text = DEFAULT_INSTRUCTIONS
if "response_format_text" not in st.session_state:
    st.session_state.response_format_text = DEFAULT_RESPONSE_FORMAT_TEXT
if "llm_timeout" not in st.session_state:
    st.session_state.llm_timeout = DEFAULT_LLM_TIMEOUT
if "vector_backend_upload" not in st.session_state:
    st.session_state.vector_backend_upload = DEFAULT_VECTOR_BACKEND
if "vector_backend_infer" not in st.session_state:
    st.session_state.vector_backend_infer = DEFAULT_VECTOR_BACKEND
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL
if "last_summary_meta" not in st.session_state:
    st.session_state.last_summary_meta = None
if "telemetry_signals" not in st.session_state:
    st.session_state.telemetry_signals = TELEMETRY_SIGNAL_DEFAULTS.copy()
if "bert_scorer_ready" not in st.session_state:
    with st.spinner("Carregando modelo de métrica semântica (BERTScore)..."):
        try:
            get_bert_scorer()
            st.session_state.bert_scorer_ready = True
        except Exception as exc:
            st.session_state.bert_scorer_ready = False
            st.warning(f"Falha ao carregar BERTScore: {exc}")
    
@st.cache_resource
def get_mqtt_queue() -> Queue:
    return Queue()


@st.cache_resource
def get_bert_scorer(model_name: str = BERT_SCORE_MODEL):
    target_model = model_name or BERT_SCORE_FALLBACK_MODEL
    try:
        return BERTScorer(lang="pt", model_type=target_model, rescale_with_baseline=True)
    except Exception as exc:
        if target_model == BERT_SCORE_FALLBACK_MODEL:
            raise
        logger.warning(
            "Falha ao inicializar BERTScore com %s (%s). Usando fallback %s.",
            target_model,
            exc,
            BERT_SCORE_FALLBACK_MODEL,
        )
        return BERTScorer(lang="pt", model_type=BERT_SCORE_FALLBACK_MODEL, rescale_with_baseline=True)

def build_mqtt_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()

mqtt_queue: "Queue[dict]" = get_mqtt_queue()

# --- MQTT Loop (Background) ---
def on_message(client, userdata, msg):
    try:
        payload_text = msg.payload.decode()
        if LOG_MQTT_EVENTS:
            logger.info("MQTT recebido | topic=%s | payload=%s", msg.topic, payload_text)
        data = json.loads(payload_text)
        mqtt_queue.put(data)
    except Exception as exc:
        logger.error("Falha ao processar mensagem MQTT: %s", exc)

@st.cache_resource
def start_mqtt():
    if not MQTT_BROKER or not MQTT_TOPIC_DATA:
        st.session_state.mqtt_error = "Variáveis de ambiente MQTT não configuradas."
        return None

    client = build_mqtt_client()
    if MQTT_USER and MQTT_PASS:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    try:
        if LOG_MQTT_EVENTS:
            logger.info(
                "Conectando ao broker %s:%s (dados=%s, comandos=%s)",
                MQTT_BROKER,
                MQTT_PORT,
                MQTT_TOPIC_DATA,
                MQTT_TOPIC_CMD,
            )
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC_DATA)
        client.on_message = on_message
        client.loop_start()
        if LOG_MQTT_EVENTS:
            logger.info("Assinatura MQTT ativa no tópico %s", MQTT_TOPIC_DATA)
        st.session_state.mqtt_error = None
        return client
    except Exception as e:
        st.session_state.mqtt_error = str(e)
        logger.error("Erro ao conectar ao broker MQTT: %s", e)
        return None

mqtt_client = start_mqtt()


def pump_mqtt_queue():
    updated = False
    while True:
        try:
            payload = mqtt_queue.get_nowait()
        except Empty:
            break
        st.session_state.telemetry = payload
        st.session_state.last_telemetry_reading = payload.copy()
        st.session_state.last_telemetry_time = datetime.now(timezone.utc).isoformat()
        if LOG_MQTT_EVENTS:
            logger.info("Atualizando dashboard com telemetria: %s", payload)
        updated = True
    return updated

def fetch_available_models(provider: str, api_key: str | None):
    params = {"provider": provider}
    if api_key:
        params["api_key"] = api_key
    try:
        response = requests.get(f"{API_URL}/llm/models", params=params, timeout=20)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(str(exc)) from exc

    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        raise RuntimeError(detail or "Erro ao listar modelos disponíveis.")

    body = response.json()
    return body.get("models", [])


def get_cached_models(provider: str, api_key: str | None):
    cache_key = f"{provider}:{api_key or 'env'}"
    if st.session_state.model_cache_key == cache_key and st.session_state.model_cache:
        return st.session_state.model_cache

    models = fetch_available_models(provider, api_key)
    st.session_state.model_cache_key = cache_key
    st.session_state.model_cache = models
    return models


def publish_command(command: str):
    if not mqtt_client:
        st.warning("Broker MQTT não conectado.")
        return False
    if not MQTT_TOPIC_CMD:
        st.warning("Tópico MQTT de comandos não configurado.")
        return False
    if LOG_MQTT_EVENTS:
        logger.info("Publicando comando MQTT | topic=%s | payload=%s", MQTT_TOPIC_CMD, command)
    mqtt_client.publish(MQTT_TOPIC_CMD, command)
    return True


def persist_experiment_log(question: str, scenario: int, response_payload: dict, llm_provider: str,
                           llm_model: str, telemetry_snapshot: dict, latency_ms: float,
                           reference_answer: Optional[str] = None,
                           accuracy: Optional[float] = None,
                           bleu: Optional[float] = None,
                           rouge_l: Optional[float] = None,
                           bert_score_f1: Optional[float] = None,
                           vector_backend: Optional[str] = None,
                           prompt_tokens: Optional[int] = None,
                           response_tokens: Optional[int] = None,
                           vector_debug: Optional[dict] = None):
    """Dispara o endpoint da API para registrar métricas exigidas no relatório."""
    log_body = {
        "question": question,
        "scenario": scenario,
        "mode_used": response_payload.get("mode_used"),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "response": response_payload.get("response", ""),
        "context_found": response_payload.get("context_found", False),
        "telemetry": telemetry_snapshot,
        "latency_ms": round(latency_ms, 2),
        "reference_answer": reference_answer,
        "accuracy": accuracy,
        "bleu": bleu,
        "rouge_l": rouge_l,
        "bert_score_f1": bert_score_f1,
        "vector_backend": vector_backend,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
    }
    if vector_debug:
        log_body["query_embedding"] = vector_debug.get("query_embedding")
        log_body["retrieved_vectors"] = vector_debug.get("retrieved")
    try:
        requests.post(f"{API_URL}/experiments/log", json=log_body, timeout=15)
    except requests.exceptions.RequestException as exc:
        st.warning(f"Falha ao gravar log experimental: {exc}")


def compute_text_metrics(candidate: str, reference: str):
    reference = (reference or "").strip()
    candidate = (candidate or "").strip()
    if not reference:
        return None, None, None, None

    accuracy = 1.0 if candidate.lower() == reference.lower() else 0.0
    bleu_score = None
    rouge_l_score = None
    bert_f1_score = None

    try:
        bleu_score = corpus_bleu([candidate], [[reference]]).score
    except Exception as exc:
        logger.warning("Falha ao calcular BLEU: %s", exc)

    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_values = scorer.score(reference, candidate)
        rouge_l_score = rouge_values["rougeL"].fmeasure * 100
    except Exception as exc:
        logger.warning("Falha ao calcular ROUGE-L: %s", exc)

    try:
        scorer = get_bert_scorer()
        _, _, f1 = scorer.score([candidate], [reference])
        bert_f1_score = float(f1.mean().item() * 100)
    except Exception as exc:
        logger.warning("Falha ao calcular BERTScore: %s", exc)

    return accuracy, bleu_score, rouge_l_score, bert_f1_score


def format_vector_preview(vector: list[float], limit: int = 16) -> str:
    if not vector:
        return "[]"
    clipped = vector[:limit]
    preview = ", ".join(f"{value:.4f}" for value in clipped)
    if len(vector) > limit:
        preview += ", ..."
    return f"[{preview}]"


def render_vector_preview(label: str, vector: list[float], limit: int = 16):
    if not vector:
        st.caption(f"{label}: sem dados disponíveis.")
        return
    st.markdown(f"**{label} (primeiras {limit} dimensões)**")
    st.code(format_vector_preview(vector, limit), language="text")
    with st.expander(f"Ver {label} completo", expanded=False):
        st.dataframe(
            {
                "dimensão": list(range(len(vector))),
                "valor": vector,
            },
            hide_index=True,
            use_container_width=True,
        )

# --- CSS Customizado para Painéis ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .diag-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURAÇÃO ---
with st.sidebar:
    st.title("🎛️ Configuração Experimental")
    
    st.subheader("1. Modelo Generativo")
    llm_provider = st.selectbox("Provedor LLM", ["groq", "gemini", "local"])
    api_key = st.text_input("API Key", type="password", help="Opcional: substitui a chave configurada no backend.")

    st.subheader("2. Seleção do Modelo")
    refresh_models = st.button("🔄 Atualizar modelos disponíveis", use_container_width=True)
    if refresh_models:
        st.session_state.model_cache_key = None
        st.session_state.model_cache = []

    model_error = None
    models_payload = []
    try:
        models_payload = get_cached_models(llm_provider, api_key or None)
    except RuntimeError as err:
        model_error = str(err)
        st.session_state.model_cache_key = None
        st.session_state.model_cache = []

    selected_model = st.session_state.llm_model_choice
    if models_payload:
        label_map = {}
        for item in models_payload:
            display = item.get("display_name") or item["id"]
            ctx = item.get("context_window") or item.get("input_token_limit")
            label_map[item["id"]] = f"{display}{f' · {ctx} tok' if ctx else ''}"
        model_ids = [item["id"] for item in models_payload]
        default_idx = model_ids.index(selected_model) if selected_model in model_ids else 0
        selected_model = st.selectbox(
            "Modelo ativo",
            options=model_ids,
            index=default_idx,
            format_func=lambda value: label_map.get(value, value)
        )
        selected_meta = next((item for item in models_payload if item["id"] == selected_model), {})
        if selected_meta.get("description"):
            st.caption(selected_meta["description"])
    else:
        selected_model = st.text_input(
            "Modelo (digite manualmente)",
            value=selected_model,
            help="Use apenas se a listagem automática falhar."
        )
        if model_error:
            st.warning(model_error)

    st.session_state.llm_model_choice = selected_model

    st.subheader("3. Contexto Estático (RAG)")
    st.session_state.vector_backend_upload = st.selectbox(
        "Backend Vetorial (indexação)",
        VECTOR_BACKEND_OPTIONS,
        index=VECTOR_BACKEND_OPTIONS.index(st.session_state.vector_backend_upload)
        if st.session_state.vector_backend_upload in VECTOR_BACKEND_OPTIONS else 0,
    )
    st.session_state.chunk_size = int(
        st.number_input(
            "Chunk size",
            min_value=200,
            max_value=4000,
            step=100,
            value=int(st.session_state.chunk_size),
        )
    )
    st.session_state.chunk_overlap = int(
        st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=2000,
            step=50,
            value=int(st.session_state.chunk_overlap),
        )
    )
    st.session_state.embedding_model = st.text_input(
        "Modelo de embedding (SentenceTransformer)",
        value=st.session_state.embedding_model,
        help=(
            "Define qual SentenceTransformer gera os vetores. Alterar o backend de indexação"
            " não troca este campo automaticamente: escolha o modelo aqui e reindexe para"
            " que FAISS/Weaviate/Pinecone recebam os novos embeddings."
        ),
    )
    # Heurística de seleção de sensores pedida no plano: escolhemos quais sinais
    # entram no prompt, facilitando estudos de ablation.
    signal_labels = {value: label for value, label in TELEMETRY_SIGNAL_OPTIONS}
    selected_signals = st.multiselect(
        "Variáveis de telemetria enviadas ao LLM",
        options=[opt[0] for opt in TELEMETRY_SIGNAL_OPTIONS],
        default=st.session_state.telemetry_signals,
        format_func=lambda value: signal_labels.get(value, value),
        help="Remova sinais que não devem entrar no prompt (ex.: ocultar corrente ao testar apenas temperatura/vibração).",
    )
    st.session_state.telemetry_signals = selected_signals or TELEMETRY_SIGNAL_DEFAULTS.copy()
    uploaded = st.file_uploader("Carregar Manual (PDF)", type="pdf")
    if uploaded and st.button("Indexar Manual"):
        with st.spinner("Vetorizando documento..."):
            files = {"file": (uploaded.name, uploaded, "application/pdf")}
            data = {
                "vector_backend": st.session_state.vector_backend_upload,
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap,
                "embedding_model": st.session_state.embedding_model,
            }
            try:
                res = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=180)
                if res.status_code != 200:
                    detail = res.json().get("detail") if res.headers.get("content-type", "").startswith("application/json") else res.text
                    st.error(f"Erro API: {detail}")
                else:
                    st.success(f"Manual indexado! Chunks: {res.json()['chunks']}")
            except Exception as e:
                st.error(f"Erro API: {e}")

    if st.button("♻️ Reprocessar base existente", help="Reindexa todos os PDFs já enviados usando o backend e parâmetros atuais."):
        with st.spinner("Reprocessando manuais existentes..."):
            payload = {
                "vector_backend": st.session_state.vector_backend_upload,
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap,
                "embedding_model": st.session_state.embedding_model,
            }
            try:
                resp = requests.post(f"{API_URL}/reindex", json=payload, timeout=240)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"{data.get('total_documents', 0)} manual(is) reprocessado(s) para {data.get('backend')} "
                        f"({data.get('total_chunks', 0)} chunks)."
                    )
                    skipped = data.get("skipped") or []
                    if skipped:
                        skipped_list = ", ".join(item.get("file", "?") for item in skipped)
                        st.warning(f"Alguns arquivos foram ignorados: {skipped_list}")
                else:
                    detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.error(f"Erro ao reprocessar: {detail}")
            except Exception as exc:
                st.error(f"Erro de conexão: {exc}")
    
    st.markdown("---")
    status_text = "🟢 Conectado" if mqtt_client else "🔴 Desconectado"
    if not mqtt_client and st.session_state.mqtt_error:
        status_text += f" — {st.session_state.mqtt_error}"
    st.info("Status do Broker: " + status_text)

    st.subheader("4. Atualização da Telemetria")
    st.session_state.auto_refresh_enabled = st.toggle(
        "Atualização automática (2s)",
        value=st.session_state.auto_refresh_enabled,
        help="Mantém o painel sincronizado sem precisar clicar em 'Atualizar Leituras'."
    )
    st.session_state.auto_refresh_interval = st.slider(
        "Intervalo (segundos)",
        min_value=1.0,
        max_value=10.0,
        step=0.5,
        value=st.session_state.auto_refresh_interval,
        help="Use valores maiores se quiser reduzir o uso de CPU.",
        disabled=not st.session_state.auto_refresh_enabled,
    )

    st.subheader("5. Relatórios de Experimentos")
    st.caption("Gere arquivos consolidados dos testes registrados em /app/data/experiment_logs.csv.")
    # Botão pedido na etapa de Experimentos: dispara o endpoint que lê o CSV e
    # exporta tabelas/gráficos prontos para o relatório científico.
    if st.button("📊 Gerar resumo automático", use_container_width=True):
        with st.spinner("Consolidando métricas e gráficos..."):
            try:
                resp = requests.post(f"{API_URL}/experiments/summarize", json={}, timeout=180)
                if resp.status_code == 200:
                    payload = resp.json()
                    st.session_state.last_summary_meta = payload
                    output_dir = payload.get("output_dir", "")
                    st.success(f"Resumo disponível em {output_dir}")
                    artifacts = payload.get("artifacts") or {}
                    if artifacts:
                        st.caption("Arquivos gerados:")
                        st.json(artifacts)
                else:
                    detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.error(f"Erro ao gerar resumo: {detail}")
            except Exception as exc:
                st.error(f"Erro de conexão: {exc}")

selected_model = st.session_state.llm_model_choice

# Atualiza telemetria com qualquer mensagem pendente do MQTT
pump_mqtt_queue()

# --- LAYOUT PRINCIPAL ---

st.title("🏭 Laboratório de IA Generativa: Diagnóstico Ciber-Físico")
st.markdown("Comparação de cenários RAG para sistemas industriais.")

# 1. PAINEL DE SENSORES (Cards)
st.subheader("📡 Monitoramento em Tempo Real (Contexto Dinâmico)")

if st.session_state.last_telemetry_time:
    st.caption(f"Última leitura: {st.session_state.last_telemetry_time}")
else:
    st.info("Aguardando dados MQTT... Certifique-se de que o simulador está publicando no tópico.")

tel = st.session_state.telemetry
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Status Máquina", tel.get("status", "N/A"))
with c2:
    temp = tel.get("temperature", 0)
    st.metric("Temperatura", f"{temp:.1f} °C", delta=f"{temp-45:.1f}", delta_color="inverse")
with c3:
    vib = tel.get("vibration", 0)
    st.metric("Vibração", f"{vib:.2f} mm/s", delta=f"{vib-2.5:.2f}", delta_color="inverse")
with c4:
    st.metric("Corrente Motor", f"{tel.get('current', 0):.1f} A")

# Botão discreto para atualizar UI (caso o stream não force repaint)
if st.button("🔄 Atualizar Leituras"):
    st.rerun()

st.markdown("---")

# 2. CONTROLE DO EXPERIMENTO
col_ctrl_1, col_ctrl_2 = st.columns([1, 2])

with col_ctrl_1:
    st.subheader("🎮 Simulador de Falhas")
    st.caption("Injete anomalias para testar o diagnóstico:")
    if st.button("✅ Operação Normal", use_container_width=True):
        publish_command("NORMAL")
    if st.button("🔥 Falha Térmica", use_container_width=True):
        publish_command("HIGH_TEMP")
    if st.button("〰️ Desbalanceamento", use_container_width=True):
        publish_command("HIGH_VIBRATION")

with col_ctrl_2:
    st.subheader("🧪 Cenário de Avaliação")
    # Radio principal usado para comparar baseline, RAG estático e dual conforme
    # solicitado no enunciado.
    scenario = st.radio(
        "Selecione o nível de contexto fornecido ao LLM:",
        [1, 2, 3],
        format_func=lambda x: {
            1: "1. Baseline (LLM Puro): Sem acesso a manuais ou sensores.",
            2: "2. RAG Estático: Acesso apenas aos Manuais PDF.",
            3: "3. Dual-Context RAG: Manuais + Dados dos Sensores em Tempo Real."
        }[x]
    )
    st.session_state.vector_backend_infer = st.selectbox(
        "Backend Vetorial (consulta)",
        VECTOR_BACKEND_OPTIONS,
        index=VECTOR_BACKEND_OPTIONS.index(st.session_state.vector_backend_infer)
        if st.session_state.vector_backend_infer in VECTOR_BACKEND_OPTIONS else 0,
        help="Define de qual base vetorial virão os chunks quando o cenário usar RAG.",
    )

st.subheader("⚙️ Prompt e Saída Estruturada")
with st.expander("Configurar base system, instruções e formato JSON", expanded=False):
    st.text_area(
        "Base System Prompt",
        value=st.session_state.base_system_text,
        key="base_system_text",
        height=150,
        help="Mensagem inicial que define o papel do LLM.",
    )
    st.text_area(
        "Instruções (uma por linha)",
        value=st.session_state.instructions_text,
        key="instructions_text",
        height=180,
        help="Será enviada antes do contexto. Use linhas separadas para cada instrução.",
    )
    st.text_area(
        "Formato de resposta (JSON)",
        value=st.session_state.response_format_text,
        key="response_format_text",
        height=220,
        help="Estrutura que o LLM deve seguir. Precisa ser um JSON válido.",
    )
    st.number_input(
        "Timeout do LLM local (segundos)",
        min_value=60,
        max_value=600,
        step=30,
        value=int(st.session_state.llm_timeout),
        key="llm_timeout",
        help="Use valores maiores se o prompt for longo.",
    )

# 3. PAINEL DE DIAGNÓSTICO
st.markdown("---")
st.subheader("🩺 Painel de Diagnóstico (LLM Output)")

query = st.text_input("Pergunta do Operador", "Qual o estado atual da máquina e recomendações de manutenção?")
show_debug = st.checkbox("Gerar logs detalhados do prompt", value=True)
log_experiments = st.checkbox("Gravar logs de experimentos (CSV)", value=False,
                              help="Armazena cada diagnóstico em /app/data/experiment_logs.csv para análise posterior.")
reference_answer = ""
if log_experiments:
    reference_answer = st.text_area(
        "Gabarito (referência para métricas)",
        value="",
        help=(
            "Cole aqui a resposta oficial (ex.: textos de docs/gabarito.md). Quando preenchido e"
            " o logging estiver ativo, a API calcula accuracy/BLEU/ROUGE comparando o LLM com o gabarito."
        ),
    )

if st.button("Gerar Relatório de Diagnóstico", type="primary"):
    if not selected_model:
        st.error("Selecione ou informe um modelo LLM antes de continuar.")
    else:
        instructions_list = [line.strip() for line in st.session_state.instructions_text.splitlines() if line.strip()]
        response_format_obj = None
        response_format_raw = st.session_state.response_format_text.strip()
        if response_format_raw:
            try:
                response_format_obj = json.loads(response_format_raw)
            except json.JSONDecodeError as exc:
                st.error(f"JSON do formato de resposta inválido: {exc}")
                response_format_obj = None
                st.stop()
        with st.spinner(f"Processando no Cenário {scenario}..."):
            start_time = time.perf_counter()
            payload = {
                "question": query,
                "scenario": scenario,
                "telemetry": st.session_state.telemetry,
                "llm_provider": llm_provider,
                "llm_model": selected_model,
                "api_key": api_key if api_key else None,
                "debug": show_debug,
                "base_system": st.session_state.base_system_text,
                "instructions": instructions_list,
                "response_format": response_format_obj,
                "llm_timeout": int(st.session_state.llm_timeout),
                "vector_backend": st.session_state.vector_backend_infer,
                "embedding_model": st.session_state.embedding_model,
                "telemetry_signals": st.session_state.telemetry_signals or TELEMETRY_SIGNAL_DEFAULTS,
            }
            # Este payload carrega tudo que o professor pode variar: provedor LLM,
            # backend vetorial, chunking, sensores selecionados e prompt template.
            
            try:
                api_timeout = max(120, int(st.session_state.llm_timeout) + 30)
                resp = requests.post(f"{API_URL}/chat", json=payload, timeout=api_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.diagnosis_history = data
                    if log_experiments:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        telemetry_snapshot = data.get("debug", {}).get("telemetry_used") or st.session_state.telemetry
                        vector_debug = data.get("vector_debug") or data.get("debug", {}).get("vector_debug")
                        accuracy = bleu = rouge_l = None
                        if reference_answer.strip():
                            accuracy, bleu, rouge_l, bert_f1 = compute_text_metrics(
                                data.get("response", ""),
                                reference_answer,
                            )
                        token_usage = data.get("token_usage", {}) or {}
                        persist_experiment_log(
                            question=query,
                            scenario=scenario,
                            response_payload=data,
                            llm_provider=llm_provider,
                            llm_model=selected_model,
                            telemetry_snapshot=telemetry_snapshot,
                            latency_ms=elapsed_ms,
                            reference_answer=reference_answer.strip() or None,
                            accuracy=accuracy,
                            bleu=bleu,
                            rouge_l=rouge_l,
                            bert_score_f1=bert_f1,
                            vector_backend=st.session_state.vector_backend_infer,
                            prompt_tokens=token_usage.get("prompt"),
                            response_tokens=token_usage.get("response"),
                            vector_debug=vector_debug,
                        )
                else:
                    detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.error(f"Falha na API: {detail}")
            except Exception as e:
                st.error(f"Erro de conexão: {e}")

# Exibição do Resultado
if st.session_state.diagnosis_history:
    res = st.session_state.diagnosis_history
    
    # Cabeçalho do Relatório
    st.markdown(f"**Modo Utilizado:** `{res['mode_used']}`")
    llm_info = res.get("debug", {}).get("llm_call") if isinstance(res, dict) else None
    if llm_info:
        st.caption(f"LLM: {llm_info.get('provider', 'n/d')} · {llm_info.get('model', 'sem modelo')}")
    if res['context_found'] and scenario > 1:
        st.success("📚 Documentação Técnica Relevante Encontrada e Utilizada.")
    elif scenario > 1:
        st.warning("⚠️ Nenhuma documentação relevante encontrada para esta consulta.")
    if res.get("vector_backend"):
        st.caption(f"Backend Vetorial: {res['vector_backend']}")
    token_info = res.get("token_usage") or {}
    if token_info:
        st.caption(f"Tokens · prompt: {token_info.get('prompt', 'n/d')} · resposta: {token_info.get('response', 'n/d')}")
        
    # Conteúdo do Diagnóstico
    with st.container(border=True):
        st.markdown(f"### 📋 Relatório Técnico")
        st.markdown(res['response'])
        
    debug_payload = res.get("debug") if isinstance(res, dict) else None
    if debug_payload:
        with st.expander("Ver logs detalhados do prompt", expanded=False):
            st.markdown("**Prompt final enviado ao LLM**")
            st.code(debug_payload.get("final_prompt", ""), language="markdown")

            st.markdown("**Trechos recuperados da base técnica**")
            chunks = debug_payload.get("retrieved_chunks") or []
            metadatas = debug_payload.get("retrieved_metadatas") or []
            if chunks:
                for idx, chunk in enumerate(chunks, start=1):
                    meta = metadatas[idx-1] if idx-1 < len(metadatas) else {}
                    source = meta.get("source", "desconhecido")
                    st.markdown(f"**Chunk {idx} · Fonte:** {source}")
                    st.write(chunk)
            else:
                st.caption("Nenhum chunk retornado pela busca vetorial.")

            st.markdown("**Telemetria utilizada no prompt**")
            st.json(debug_payload.get("telemetry_used", {}))

            if debug_payload.get("prompt_sections"):
                st.markdown("**Seções do prompt (para inspeção rápida)**")
                st.json(debug_payload["prompt_sections"])

            vector_debug = res.get("vector_debug") or debug_payload.get("vector_debug")
            if vector_debug:
                with st.expander("Embeddings e similaridade (cosine)", expanded=False):
                    query_vec = vector_debug.get("query_embedding") or []
                    model_name = vector_debug.get("embedding_model", "n/d")
                    st.caption(
                        f"Modelo de embedding: {model_name} · Dimensão do vetor: {len(query_vec) if query_vec else 'n/d'}"
                    )
                    render_vector_preview("Vetor da pergunta", query_vec)

                    retrieved_items = vector_debug.get("retrieved") or []
                    if not retrieved_items:
                        st.caption("Nenhum chunk vetorial disponível para esta consulta.")
                    else:
                        st.markdown("**Chunks recuperados (vetores completos e similaridade)**")
                        for item in retrieved_items:
                            idx = item.get("index")
                            chunk_label = idx + 1 if isinstance(idx, int) else "?"
                            source = item.get("source") or "desconhecido"
                            similarity = item.get("similarity")
                            if isinstance(similarity, (int, float)):
                                similarity_str = f"{similarity:.4f}"
                            else:
                                similarity_str = "n/d"
                            st.markdown(f"**Chunk {chunk_label} · Fonte:** {source}")
                            st.caption(f"Similaridade (cosseno): {similarity_str}")
                            if item.get("chunk_preview"):
                                st.write(item["chunk_preview"])
                            render_vector_preview(f"Vetor do chunk {chunk_label}", item.get("embedding", []))

    st.caption("Este resultado deve ser comparado com o 'Ground Truth' para avaliação experimental.")

# Atualização automática do dashboard (opcional)
if st.session_state.auto_refresh_enabled:
    time.sleep(st.session_state.auto_refresh_interval)
    st.rerun()