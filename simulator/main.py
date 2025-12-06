import time
import json
import random
import os
import paho.mqtt.client as mqtt

# Configurações
BROKER = os.getenv("MQTT_BROKER_ADDRESS", "test.mosquitto.org")
PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
USER = os.getenv("MQTT_USERNAME")
PASS = os.getenv("MQTT_PASSWORD")
TOPIC_DATA = os.getenv("MQTT_TOPIC_SENSORS", "industrial/lathe/sensors")
TOPIC_CMD = os.getenv("MQTT_TOPIC_COMMANDS", "industrial/lathe/commands")

# Estado Inicial
state = {
    "status": "NORMAL",
    "temperature": 45.0,
    "vibration": 2.5,
    "current": 12.0
}

mode = "NORMAL"

def on_connect(client, userdata, flags, rc):
    print(f"Simulador Online (RC: {rc})")
    client.subscribe(TOPIC_CMD)

def on_message(client, userdata, msg):
    global mode
    cmd = msg.payload.decode().upper()
    print(f"-> Comando: {cmd}")
    if "NORMAL" in cmd: mode = "NORMAL"
    elif "HIGH_TEMP" in cmd: mode = "OVERHEAT"
    elif "HIGH_VIBRATION" in cmd: mode = "UNBALANCED"

client = mqtt.Client()
if USER and PASS:
    client.username_pw_set(USER, PASS)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER, PORT, 60)
    client.loop_start()
except Exception as e:
    print(f"Erro Conexão Broker: {e}")
    # Fallback para loop local sem MQTT (para debug)
    pass

print("Iniciando geração de dados...")
while True:
    # Física Simplificada
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
    try:
        payload = json.dumps(state)
        client.publish(TOPIC_DATA, payload)
        print(f"Status: {mode} | T: {state['temperature']:.1f} | V: {state['vibration']:.1f}")
    except:
        pass
        
    time.sleep(2)