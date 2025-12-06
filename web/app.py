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

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
logger = logging.getLogger("web-app")

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

st.set_page_config(page_title="Industrial Dual-RAG Lab", layout="wide")

# --- Estado ---
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

def build_mqtt_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()

mqtt_queue: "Queue[dict]" = Queue()

# --- MQTT Loop (Background) ---
def on_message(client, userdata, msg):
    try:
        payload_text = msg.payload.decode()
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
    logger.info("Publicando comando MQTT | topic=%s | payload=%s", MQTT_TOPIC_CMD, command)
    mqtt_client.publish(MQTT_TOPIC_CMD, command)
    return True


def persist_experiment_log(question: str, scenario: int, response_payload: dict, llm_provider: str,
                           llm_model: str, telemetry_snapshot: dict, latency_ms: float):
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
    }
    try:
        requests.post(f"{API_URL}/experiments/log", json=log_body, timeout=15)
    except requests.exceptions.RequestException as exc:
        st.warning(f"Falha ao gravar log experimental: {exc}")

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
    uploaded = st.file_uploader("Carregar Manual (PDF)", type="pdf")
    if uploaded and st.button("Indexar Manual"):
        with st.spinner("Vetorizando documento..."):
            files = {"file": (uploaded.name, uploaded, "application/pdf")}
            try:
                res = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                if res.status_code != 200:
                    detail = res.json().get("detail") if res.headers.get("content-type", "").startswith("application/json") else res.text
                    st.error(f"Erro API: {detail}")
                else:
                    st.success(f"Manual indexado! Chunks: {res.json()['chunks']}")
            except Exception as e:
                st.error(f"Erro API: {e}")
    
    st.markdown("---")
    status_text = "🟢 Conectado" if mqtt_client else "🔴 Desconectado"
    if not mqtt_client and st.session_state.mqtt_error:
        status_text += f" — {st.session_state.mqtt_error}"
    st.info("Status do Broker: " + status_text)

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
    scenario = st.radio(
        "Selecione o nível de contexto fornecido ao LLM:",
        [1, 2, 3],
        format_func=lambda x: {
            1: "1. Baseline (LLM Puro): Sem acesso a manuais ou sensores.",
            2: "2. RAG Estático: Acesso apenas aos Manuais PDF.",
            3: "3. Dual-Context RAG: Manuais + Dados dos Sensores em Tempo Real."
        }[x]
    )

# 3. PAINEL DE DIAGNÓSTICO
st.markdown("---")
st.subheader("🩺 Painel de Diagnóstico (LLM Output)")

query = st.text_input("Pergunta do Operador", "Qual o estado atual da máquina e recomendações de manutenção?")
show_debug = st.checkbox("Gerar logs detalhados do prompt", value=True)
log_experiments = st.checkbox("Gravar logs de experimentos (CSV)", value=False,
                              help="Armazena cada diagnóstico em /app/data/experiment_logs.csv para análise posterior.")

if st.button("Gerar Relatório de Diagnóstico", type="primary"):
    if not selected_model:
        st.error("Selecione ou informe um modelo LLM antes de continuar.")
    else:
        with st.spinner(f"Processando no Cenário {scenario}..."):
            start_time = time.perf_counter()
            payload = {
                "question": query,
                "scenario": scenario,
                "telemetry": st.session_state.telemetry,
                "llm_provider": llm_provider,
                "llm_model": selected_model,
                "api_key": api_key if api_key else None,
                "debug": show_debug
            }
            
            try:
                resp = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.diagnosis_history = data
                    if log_experiments:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        telemetry_snapshot = data.get("debug", {}).get("telemetry_used") or st.session_state.telemetry
                        persist_experiment_log(
                            question=query,
                            scenario=scenario,
                            response_payload=data,
                            llm_provider=llm_provider,
                            llm_model=selected_model,
                            telemetry_snapshot=telemetry_snapshot,
                            latency_ms=elapsed_ms,
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

    st.caption("Este resultado deve ser comparado com o 'Ground Truth' para avaliação experimental.")