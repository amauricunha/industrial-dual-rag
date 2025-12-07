"""Simulador MQTT que injeta telemetria sintética para o estudo de caso do torno."""

import time
import json
import random
import os
import logging
import paho.mqtt.client as mqtt

# Configurações
BROKER = os.getenv("MQTT_BROKER_ADDRESS", "test.mosquitto.org")
PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
USER = os.getenv("MQTT_USERNAME")
PASS = os.getenv("MQTT_PASSWORD")
TOPIC_DATA = os.getenv("MQTT_TOPIC_SENSORS", "industrial/lathe/sensors")
TOPIC_CMD = os.getenv("MQTT_TOPIC_COMMANDS", "industrial/lathe/commands")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
logger = logging.getLogger("simulator")

# Estado Inicial
# Mantemos quatro sinais (status, temperatura, vibração, corrente) pois eles
# alimentam o seletor de sensores e o cenário Dual do RAG.
state = {
    "status": "NORMAL",
    "temperature": 45.0,
    "vibration": 2.5,
    "current": 12.0
}

mode = "NORMAL"

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Conectado ao broker %s:%s", BROKER, PORT)
        client.subscribe(TOPIC_CMD)
        logger.info("Assinando comandos em %s", TOPIC_CMD)
    else:
        logger.error("Falha na conexão MQTT (rc=%s)", rc)

def on_message(client, userdata, msg):
    global mode
    cmd = msg.payload.decode().upper()
    logger.info("Comando recebido | topic=%s | payload=%s", msg.topic, cmd)
    if "NORMAL" in cmd: mode = "NORMAL"
    elif "HIGH_TEMP" in cmd: mode = "OVERHEAT"
    elif "HIGH_VIBRATION" in cmd: mode = "UNBALANCED"

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()
if USER and PASS:
    client.username_pw_set(USER, PASS)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER, PORT, 60)
    client.loop_start()
except Exception as e:
    logger.error("Erro ao conectar ao broker: %s", e)
    # Fallback para loop local sem MQTT (para debug)
    pass

logger.info("Iniciando geração de dados. Publicando em %s", TOPIC_DATA)
while True:
    # Física Simplificada: cada modo altera os sinais para criar casos de teste
    # repetíveis (baseline vs. falha), conforme exigido no enunciado.
    if mode == "NORMAL":
        state["temperature"] = 45.0 + random.uniform(-1, 1)
        state["vibration"] = 2.5 + random.uniform(-0.2, 0.2)
        state["status"] = "OPERATIONAL"
    elif mode == "OVERHEAT":
        state["temperature"] = min(state["temperature"] + 2.0, 110.0) # Sobe rápido
        state["vibration"] = 2.8 + random.uniform(-0.2, 0.2)
        state["status"] = "WARNING_TEMP"
    elif mode == "UNBALANCED":
        state["vibration"] = min(state["vibration"] + 1.5, 20.0)
        state["temperature"] += 0.1
        state["status"] = "CRITICAL_VIB"
    
    state["current"] = 12.0 + (state["vibration"] * 0.5) # Corrente sobe com vibração

    # Envio
    payload = json.dumps(state)
    result = client.publish(TOPIC_DATA, payload)
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        logger.info(
            "Telemetria publicada | topic=%s | payload=%s",
            TOPIC_DATA,
            payload,
        )
    else:
        logger.error("Falha ao publicar telemetria (rc=%s)", result.rc)
        
    time.sleep(2)