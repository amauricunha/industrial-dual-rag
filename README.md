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
* Vector Database configurável (ChromaDB, FAISS, Weaviate e Pinecone): armazena embeddings dos manuais técnicos (Contexto Estático).
* LLM Gateway: Conecta-se a modelos externos (Groq, Gemini) ou locais.
* Lógica de Experimento: Monta o prompt dinamicamente baseada no cenário escolhido (1, 2 ou 3), permitindo ajuste de `base_system`, instruções e formato de resposta JSON.
* Métricas automáticas (accuracy, BLEU, ROUGE-L, latência e tokens) persistidas em CSV quando o registro de experimentos está ativo.

3. Interface do Operador (/web):

* Frontend Streamlit: Dashboard para visualização de dados.
* Cliente MQTT: Assina tópicos de sensores para exibir dados em tempo real.
* Controle Experimental: Permite upload de PDFs, escolha do backend vetorial, parâmetros de chunking/embedding, injeção de falhas, edição do prompt base/instruções/JSON e seleção de cenário RAG.
* Registro de Experimentos: coleta métricas e gabaritos opcionais; notebook em `/notebooks/experiment_summary.ipynb` consolida os resultados em tabelas e gráficos.

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
OLLAMA_CHAT_TIMEOUT=180

# Vetorização / RAG
VECTOR_BACKEND_DEFAULT=chroma  # opções: chroma, faiss, weaviate, pinecone
CHUNK_SIZE_DEFAULT=1000
CHUNK_OVERLAP_DEFAULT=200
EMBEDDING_MODEL_DEFAULT=all-MiniLM-L6-v2
FAISS_INDEX_DIR=/app/data/faiss_index

# Relatórios de experimentos
SUMMARY_OUTPUT_DIR=/app/data/summaries
SUMMARY_MAX_RECENT=50

# Weaviate (opcional)
WEAVIATE_URL=http://weaviate:8080  # container local já incluso no docker-compose
WEAVIATE_API_KEY=                  # deixe vazio para uso local sem autenticação
WEAVIATE_CLASS=IndustrialManual

# Pinecone (opcional)
PINECONE_API_KEY=sua_chave_pinecone
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=industrial-dual-rag
PINECONE_NAMESPACE=default
PINECONE_DIMENSION=384
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
````

2. Suba os contêineres:

````
docker-compose up --build
````

3. Acesse a interface web:

* URL: `http://localhost:8501`

### Reprocessar base ao trocar o backend vetorial

- Os PDFs enviados ficam salvos em `/app/data/uploads` e são reutilizados para qualquer backend habilitado (Chroma, FAISS, Weaviate ou Pinecone).
- Ao mudar o backend no painel lateral, clique no botão `♻️ Reprocessar base existente` para reindexar automaticamente todos os manuais já carregados com os novos parâmetros de chunking/embedding.
- Esse passo evita subir os PDFs novamente e garante que o backend recém-selecionado receba os mesmos documentos antes de executar consultas.
- Para Pinecone ou Weaviate externos, certifique-se de preencher as variáveis no `.env` antes de reprocessar para evitar erros de autenticação.

### Persistência de dados e relatórios

- O serviço `api` monta `./data/api` (host) em `/app/data`, concentrando `experiment_logs.csv`, PDFs processados e os resumos gerados em `SUMMARY_OUTPUT_DIR`. Assim, você pode abrir os CSV/HTML fora do Docker sem depender da UI.
- O contêiner do Weaviate escreve em `./data/weaviate`; mantenha essa pasta para preservar o índice local entre rebuilds.
- Antes de executar `docker-compose up`, crie as pastas necessárias: `mkdir -p data/api data/weaviate`.

## Protocolo de Experimento

Para reproduzir os resultados do relatório científico:

### Passo 1: Preparação

1. Na barra lateral, selecione o LLM (Recomendado: Groq/Llama3 para velocidade).

2. Faça upload do arquivo manual_torno.pdf (disponível na pasta /docs ou use um genérico).

3. Clique em "Indexar Manual".

4. Opcional: ajuste o seletor "Variáveis de telemetria enviadas ao LLM" para limitar quais sinais (temperatura, vibração, corrente, status) entram no prompt dos cenários com contexto dinâmico.

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

4. Registro e Consolidação:

* Ative o checkbox "Gravar logs de experimentos", informe um gabarito (quando houver) e execute diagnósticos.
* As métricas são gravadas em `/app/data/experiment_logs.csv`. Após capturar os cenários desejados, clique no botão "📊 Gerar resumo automático" da barra lateral para consolidar CSVs e gráficos em `SUMMARY_OUTPUT_DIR` (padrão: `/app/data/summaries`). Se preferir inspeção manual, continue usando o notebook `notebooks/experiment_summary.ipynb`, que consome os mesmos arquivos.

## Limitações Operacionais

- **Broker MQTT público:** o padrão (`test.mosquitto.org`) não oferece SLA, podendo sofrer quedas ou limitação de mensagens. Para medições consistentes, substitua por um broker privado (Eclipse Mosquitto local ou serviço gerenciado) e atualize as variáveis `MQTT_*` no `.env`.
- **Dependência de APIs externas:** provedores como Groq e Google impõem limites de taxa e de tokens; latências ou erros 429 impactam diretamente o tempo de diagnóstico. Para cenários offline, mantenha Ollama com o modelo baixado previamente e ajuste `OLLAMA_CHAT_TIMEOUT` conforme o tamanho do prompt.
- **Estado dos backends vetoriais:** cada backend mantém seu próprio índice; ao alternar entre Chroma/FAISS/Weaviate/Pinecone é obrigatório reprocessar os PDFs (botão `♻️ Reprocessar base existente`). Serviços externos ainda exigem conectividade estável e chaves válidas.
- **Persistência e espaço em disco:** logs, uploads e resumos ficam em `./data/api`. O volume cresce com novos experimentos; faça limpeza periódica ou mova os arquivos gerados para armazenamento frio. O índice do Weaviate consome `./data/weaviate` e pode ultrapassar centenas de MB dependendo da base.

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
├── notebooks/          # Notebook para consolidação de métricas
└── docker-compose.yml  # Orquestração
````

## Tecnologias Utilizadas

* LLMs: Llama3 (via Groq), Gemini Pro.

* RAG: ChromaDB, FAISS, Weaviate, Pinecone (Vector Stores), Sentence-Transformers.

* Backend: FastAPI, Python.

* Frontend: Streamlit.

* IoT: Protocolo MQTT (Paho MQTT), Eclipse Mosquitto.


**Autores**: Amauri Cunha, Yessica Maria Valencia Lemos
**Data**: 06 de Dezembro 2025