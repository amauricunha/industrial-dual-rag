"""Simulador MQTT que injeta telemetria sintética para o estudo de caso do torno."""

import time
import json
import random
import os
import logging
from pathlib import Path
import paho.mqtt.client as mqtt

# Configurações
MQTT_CONFIG_PATH = Path(os.getenv("MQTT_CONFIG_PATH", "/app/data/mqtt_config.json"))


def load_mqtt_runtime_config() -> dict:
    env_config = {
        "broker": os.getenv("MQTT_BROKER_ADDRESS", "test.mosquitto.org"),
        "port": int(os.getenv("MQTT_BROKER_PORT", 1883)),
        "use_auth": bool(os.getenv("MQTT_USERNAME") and os.getenv("MQTT_PASSWORD")),
        "username": os.getenv("MQTT_USERNAME", ""),
        "password": os.getenv("MQTT_PASSWORD", ""),
        "topic_sensors": os.getenv("MQTT_TOPIC_SENSORS", "industrial/lathe/sensors"),
        "topic_commands": os.getenv("MQTT_TOPIC_COMMANDS", "industrial/lathe/commands"),
    }
    try:
        if MQTT_CONFIG_PATH.exists():
            with MQTT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                env_config.update(payload)
                logger.info("Configuração MQTT carregada de %s", MQTT_CONFIG_PATH)
    except Exception as exc:
        logger.warning("Falha ao ler configuração MQTT (%s): %s", MQTT_CONFIG_PATH, exc)

    try:
        env_config["port"] = int(env_config.get("port", 1883))
    except Exception:
        env_config["port"] = 1883
    env_config["broker"] = str(env_config.get("broker", "test.mosquitto.org")).strip() or "test.mosquitto.org"
    env_config["topic_sensors"] = str(env_config.get("topic_sensors", "industrial/lathe/sensors")).strip() or "industrial/lathe/sensors"
    env_config["topic_commands"] = str(env_config.get("topic_commands", "industrial/lathe/commands")).strip() or "industrial/lathe/commands"
    env_config["username"] = str(env_config.get("username", ""))
    env_config["password"] = str(env_config.get("password", ""))
    env_config["use_auth"] = bool(env_config.get("use_auth"))
    return env_config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
logger = logging.getLogger("simulator")

runtime_mqtt = load_mqtt_runtime_config()
BROKER = runtime_mqtt["broker"]
PORT = runtime_mqtt["port"]
USER = runtime_mqtt["username"]
PASS = runtime_mqtt["password"]
USE_AUTH = runtime_mqtt["use_auth"]
TOPIC_DATA = runtime_mqtt["topic_sensors"]
TOPIC_CMD = runtime_mqtt["topic_commands"]
ACTIVE_CONFIG_SIGNATURE = json.dumps(runtime_mqtt, sort_keys=True)

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
last_connect_attempt = 0.0


def reconnect_client(client, config: dict, reason: str = "") -> bool:
    global BROKER, PORT, USER, PASS, USE_AUTH, TOPIC_DATA, TOPIC_CMD, ACTIVE_CONFIG_SIGNATURE

    BROKER = config["broker"]
    PORT = config["port"]
    USER = config["username"]
    PASS = config["password"]
    USE_AUTH = config["use_auth"]
    TOPIC_DATA = config["topic_sensors"]
    TOPIC_CMD = config["topic_commands"]
    ACTIVE_CONFIG_SIGNATURE = json.dumps(config, sort_keys=True)

    try:
        client.loop_stop()
    except Exception:
        pass
    try:
        client.disconnect()
    except Exception:
        pass

    if USE_AUTH and USER:
        client.username_pw_set(USER, PASS)
    else:
        client.username_pw_set(None, None)

    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        logger.info(
            "Conexão MQTT solicitada | broker=%s:%s | topic_data=%s | topic_cmd=%s | motivo=%s",
            BROKER,
            PORT,
            TOPIC_DATA,
            TOPIC_CMD,
            reason or "n/a",
        )
        return True
    except Exception as exc:
        logger.error("Erro ao conectar ao broker: %s", exc)
        return False

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
client.on_connect = on_connect
client.on_message = on_message

reconnect_client(client, runtime_mqtt, reason="startup")

logger.info("Iniciando geração de dados. Publicando em %s", TOPIC_DATA)
while True:
    latest_cfg = load_mqtt_runtime_config()
    latest_sig = json.dumps(latest_cfg, sort_keys=True)
    now = time.time()
    if latest_sig != ACTIVE_CONFIG_SIGNATURE:
        reconnect_client(client, latest_cfg, reason="config_changed")
        logger.info("Configuração MQTT atualizada via arquivo compartilhado.")
    elif not client.is_connected() and now - last_connect_attempt >= 5:
        reconnect_client(client, latest_cfg, reason="retry")
        last_connect_attempt = now

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