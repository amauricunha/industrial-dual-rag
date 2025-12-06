# Industrial Dual-Context RAG: Cyber-Physical Diagnosis System

Este repositório contém a implementação de um sistema de Geração Aumentada por Recuperação (RAG) aplicado ao contexto industrial. O projeto investiga o impacto da fusão de Contexto Estático (Manuais Técnicos PDF) e Contexto Dinâmico (Telemetria IoT via MQTT) na precisão de diagnósticos gerados por Grandes Modelos de Linguagem (LLMs).

Projeto desenvolvido como requisito para a disciplina de IA Generativa (Mestrado em Engenharia de Automação e Sistemas).

## Arquitetura do Sistema

O sistema é composto por três módulos principais orquestrados via Docker:

1. Simulador IoT (/simulator):

* Simula uma máquina industrial (ex: Torno CNC).
* Gera dados sintéticos de vibração, temperatura e corrente.
* Permite injeção de falhas (Superaquecimento, Desbalanceamento) via comandos MQTT.

2. API RAG (/api):

* Backend FastAPI: Gerencia o pipeline de inferência.
* Vector Database (ChromaDB): Armazena embeddings dos manuais técnicos (Contexto Estático).
* LLM Gateway: Conecta-se a modelos externos (Groq, Gemini) ou locais.
* Lógica de Experimento: Monta o prompt dinamicamente baseada no cenário escolhido (1, 2 ou 3).

3. Interface do Operador (/web):

* Frontend Streamlit: Dashboard para visualização de dados.
* Cliente MQTT: Assina tópicos de sensores para exibir dados em tempo real.
* Controle Experimental: Permite upload de PDFs, injeção de falhas e seleção de cenário RAG.

## Fluxo de Dados

1. Loop de Telemetria (Tempo Real)
```
[Simulador] --(JSON via MQTT)--> [Broker] --(Subscrição)--> [Interface Web]
```
* O simulador publica dados a cada 2 segundos.
* A interface web atualiza o estado da sessão (Session State) com a última leitura.

2. Loop de Diagnóstico (On-Demand)

````
[Usuário] + [Estado Atual] --> [API] --> [ChromaDB] + [LLM] --> [Diagnóstico]
````

Quando o usuário solicita um diagnóstico, o fluxo depende do cenário:

* Cenário 1 (Baseline): Prompt = Pergunta

* Cenário 2 (RAG Estático): Prompt = Pergunta + Trechos do PDF

* Cenário 3 (Dual Context): Prompt = Pergunta + Trechos do PDF + Telemetria Atual (JSON)

## Como Executar

### Pré-requisitos

* Docker e Docker Compose instalados.

* (Opcional) Chave de API da Groq ou Google AI Studio.

### Configuração

1. Renomeie ou crie o arquivo .env na raiz:

````
# Broker MQTT (Público para testes ou Local)
MQTT_BROKER_ADDRESS=test.mosquitto.org
MQTT_BROKER_PORT=1883

# Chaves de API (Necessário para a inferência real)
GROQ_API_KEY=sua_chave_aqui
GOOGLE_API_KEY=sua_chave_aqui
````

2. Suba os contêineres:

````
docker-compose up --build
````

3. Acesse a interface web:

* URL: `http://localhost:8501`

## Protocolo de Experimento

Para reproduzir os resultados do relatório científico:

### Passo 1: Preparação

1. Na barra lateral, selecione o LLM (Recomendado: Groq/Llama3 para velocidade).

2. Faça upload do arquivo manual_torno.pdf (disponível na pasta /docs ou use um genérico).

3. Clique em "Indexar Manual".

### Passo 2: Execução do Teste

1. Estado Normal:

* Deixe o simulador em "Operação Normal".

* Selecione Cenário 3.

* Pergunte: "Qual o estado da máquina?".

* Resultado Esperado: O LLM deve informar que os parâmetros estão nominais.

2. Injeção de Falha:

* Clique no botão "🔥 Falha Térmica".

* Aguarde a temperatura no painel subir acima de 90°C.

3. Comparação de Cenários (Ablation Study):

* Cenário 1 (Baseline): Pergunte "O que devo fazer?". O LLM não saberá da temperatura alta.

* Cenário 3 (Dual Context): Pergunte "O que devo fazer?". O LLM deve detectar o superaquecimento (via Telemetria) e citar o procedimento de resfriamento (via Manual PDF).

## Estrutura de Arquivos

````
.
├── api/                # Backend FastAPI e Lógica RAG
│   ├── main.py         # Endpoints e construção de prompts
│   └── Dockerfile
├── web/                # Frontend Streamlit
│   ├── app.py          # Dashboard e Cliente MQTT
│   └── Dockerfile
├── simulator/          # Script Python de Simulação IoT
│   └── main.py
├── data_storage/       # Persistência do ChromaDB (Gerado automaticamente)
└── docker-compose.yml  # Orquestração
````

## Tecnologias Utilizadas

* LLMs: Llama3 (via Groq), Gemini Pro.

* RAG: ChromaDB (Vector Store), Sentence-Transformers.

* Backend: FastAPI, Python.

* Frontend: Streamlit.

* IoT: Protocolo MQTT (Paho MQTT), Eclipse Mosquitto.


**Autores**: Amauri Cunha, Yessica Maria Valencia Lemos
**Data**: 06 de Dezembro 2025