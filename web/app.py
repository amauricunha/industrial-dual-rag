import streamlit as st
import requests
import json
import time
import os
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# --- Setup ---
API_URL = os.getenv("API_URL", "http://api:8000")
MQTT_BROKER = os.getenv("MQTT_BROKER_ADDRESS")
MQTT_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
MQTT_TOPIC_DATA = os.getenv("MQTT_TOPIC_SENSORS")
MQTT_TOPIC_CMD = os.getenv("MQTT_TOPIC_COMMANDS")
MQTT_USER = os.getenv("MQTT_USERNAME")
MQTT_PASS = os.getenv("MQTT_PASSWORD")

st.set_page_config(page_title="Industrial Dual-RAG Lab", layout="wide")

# --- Estado ---
if "telemetry" not in st.session_state:
    st.session_state.telemetry = {"temperature": 0, "vibration": 0, "current": 0, "status": "OFFLINE"}
if "diagnosis_history" not in st.session_state:
    st.session_state.diagnosis_history = None

# --- MQTT Loop (Background) ---
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        st.session_state.telemetry = data
    except: pass

@st.cache_resource
def start_mqtt():
    client = mqtt.Client()
    if MQTT_USER and MQTT_PASS:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC_DATA)
        client.on_message = on_message
        client.loop_start()
        return client
    except Exception as e:
        return None

mqtt_client = start_mqtt()

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
    api_key = st.text_input("API Key", type="password")

    st.subheader("2. Contexto Estático (RAG)")
    uploaded = st.file_uploader("Carregar Manual (PDF)", type="pdf")
    if uploaded and st.button("Indexar Manual"):
        with st.spinner("Vetorizando documento..."):
            files = {"file": (uploaded.name, uploaded, "application/pdf")}
            try:
                res = requests.post(f"{API_URL}/upload", files=files)
                st.success(f"Manual indexado! Chunks: {res.json()['chunks']}")
            except Exception as e:
                st.error(f"Erro API: {e}")
    
    st.markdown("---")
    st.info("Status do Broker: " + ("🟢 Conectado" if mqtt_client else "🔴 Desconectado"))

# --- LAYOUT PRINCIPAL ---

st.title("🏭 Laboratório de IA Generativa: Diagnóstico Ciber-Físico")
st.markdown("Comparação de cenários RAG para sistemas industriais.")

# 1. PAINEL DE SENSORES (Cards)
st.subheader("📡 Monitoramento em Tempo Real (Contexto Dinâmico)")

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
        mqtt_client.publish(MQTT_TOPIC_CMD, "NORMAL")
    if st.button("🔥 Falha Térmica", use_container_width=True):
        mqtt_client.publish(MQTT_TOPIC_CMD, "HIGH_TEMP")
    if st.button("〰️ Desbalanceamento", use_container_width=True):
        mqtt_client.publish(MQTT_TOPIC_CMD, "HIGH_VIBRATION")

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

if st.button("Gerar Relatório de Diagnóstico", type="primary"):
    with st.spinner(f"Processando no Cenário {scenario}..."):
        payload = {
            "question": query,
            "scenario": scenario,
            "telemetry": st.session_state.telemetry,
            "llm_provider": llm_provider,
            "api_key": api_key if api_key else None
        }
        
        try:
            resp = requests.post(f"{API_URL}/chat", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.diagnosis_history = data
            else:
                st.error("Falha na API")
        except Exception as e:
            st.error(f"Erro de conexão: {e}")

# Exibição do Resultado
if st.session_state.diagnosis_history:
    res = st.session_state.diagnosis_history
    
    # Cabeçalho do Relatório
    st.markdown(f"**Modo Utilizado:** `{res['mode_used']}`")
    if res['context_found'] and scenario > 1:
        st.success("📚 Documentação Técnica Relevante Encontrada e Utilizada.")
    elif scenario > 1:
        st.warning("⚠️ Nenhuma documentação relevante encontrada para esta consulta.")
        
    # Conteúdo do Diagnóstico
    with st.container(border=True):
        st.markdown(f"### 📋 Relatório Técnico")
        st.markdown(res['response'])
        
    st.caption("Este resultado deve ser comparado com o 'Ground Truth' para avaliação experimental.")