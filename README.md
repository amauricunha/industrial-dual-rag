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
# Broker MQTT (público ou privado)
MQTT_BROKER_ADDRESS=<ip ou endereco>
MQTT_BROKER_PORT=<porta>
MQTT_USERNAME=<login>
MQTT_PASSWORD=<senha>
MQTT_TOPIC_SENSORS=industrial/lathe/sensors
MQTT_TOPIC_COMMANDS=industrial/lathe/commands
LOG_MQTT_EVENTS=false

# Chaves de API (Necessário para a inferência real)
GROQ_API_KEY=sua_chave_aqui
GOOGLE_API_KEY=sua_chave_aqui
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_CHAT_TIMEOUT=600
HUGGING_FACE_TOKEN=

# Vetorização / RAG
VECTOR_BACKEND_DEFAULT=chroma  # opções: chroma, faiss, weaviate, pinecone
CHUNK_SIZE_DEFAULT=1000
CHUNK_OVERLAP_DEFAULT=200
EMBEDDING_MODEL_DEFAULT=all-MiniLM-L6-v2
FAISS_INDEX_DIR=/app/data/faiss_index
CHROMA_DB_PATH=/app/data/chromadb

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

### Modelo de embedding × backend vetorial

- O modelo de embedding padrão é `all-MiniLM-L6-v2` (Sentence-Transformers). Ele é configurável tanto pelo `.env` (`EMBEDDING_MODEL_DEFAULT`) quanto pelo campo **"Modelo de embedding"** no Streamlit.
- Alterar o backend de indexação **não** troca automaticamente o modelo de embedding; a escolha do SentenceTransformer é global. Se você mudar o modelo, reexecute `♻️ Reprocessar base existente` para gerar vetores com o novo encoder para Chroma/FAISS/Weaviate/Pinecone.
- Chroma usa o `embedding_function` definido no backend, mas FAISS/Weaviate/Pinecone carregam explicitamente o modelo informado, garantindo comparações justas entre backends mesmo quando você alterna entre eles.
- Documentamos essa decisão para atender ao critério experimental do professor: o estudo mantém o mesmo encoder enquanto troca apenas o armazenamento vetorial, isolando o efeito do backend.

#### Outros encoders possíveis (SentenceTransformers):

- all-mpnet-base-v2 – melhor recall geral, porém vetores de 768 dimensões (mais pesados).
- multi-qa-mpnet-base-dot-v1 – otimizado para perguntas/respostas, boa escolha para manuais.
- gte-large (GEMMA Text Embedding) – alternativa recente, 1024 dimensões.
paraphrase-multilingual-mpnet-base-v2 – se precisar suportar PT/EN simultaneamente.

Basta digitar o nome exato no campo “Modelo de embedding” da UI (desde que o pacote sentence-transformers o suporte) e reindexar.

### Persistência de dados e relatórios

- O serviço `api` monta `./data/api` (host) em `/app/data`, concentrando `experiment_logs.csv`, PDFs processados e os resumos gerados em `SUMMARY_OUTPUT_DIR`. Assim, você pode abrir os CSV/HTML fora do Docker sem depender da UI.
- O contêiner do Weaviate escreve em `./data/weaviate`; mantenha essa pasta para preservar o índice local entre rebuilds.
- Antes de executar `docker-compose up`, crie as pastas necessárias: `mkdir -p data/api data/weaviate`.

## Protocolo de Experimento

Para reproduzir os resultados do relatório científico:

### Passo 1: Preparação

1. Na barra lateral, selecione o LLM (Recomendado: Groq/Llama3 para velocidade).
Recommended Trio

    - Groq → llama-3.3-70b-versatile · 131072 tok
    - Gemini → Gemini 2.5 Flash · 1048576 tok
    - Local → Ollama llama3.2:3b (ja carregado pelo docker-compose)

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

### Gabaritos de referência

- O campo "Gabarito (referência para métricas)" compara a resposta do LLM com um texto oficial para calcular accuracy/BLEU/ROUGE.
- Utilize o arquivo `docs/gabarito.md`, que traz respostas derivadas do **Manual de Operação e Manutenção – ROMI T 240** para os cenários Normal, Falha Térmica e Desbalanceamento. Basta copiar o trecho do cenário correspondente para o campo da UI antes de rodar o teste.
- Se adicionar novas falhas ou traduzir o manual, edite o arquivo mantendo a estrutura "Estado geral → Evidências → Ações" para preservar a consistência estatística.

## Limitações Operacionais

- **Broker MQTT público:** o padrão (`test.mosquitto.org`) não oferece SLA, podendo sofrer quedas ou limitação de mensagens. Para medições consistentes, substitua por um broker privado (Eclipse Mosquitto local ou serviço gerenciado) e atualize as variáveis `MQTT_*` no `.env`.
- **Dependência de APIs externas:** provedores como Groq e Google impõem limites de taxa e de tokens; latências ou erros 429 impactam diretamente o tempo de diagnóstico. Para cenários offline, mantenha Ollama com o modelo baixado previamente e ajuste `OLLAMA_CHAT_TIMEOUT` conforme o tamanho do prompt.
- **Estado dos backends vetoriais:** cada backend mantém seu próprio índice; ao alternar entre Chroma/FAISS/Weaviate/Pinecone é obrigatório reprocessar os PDFs (botão `♻️ Reprocessar base existente`). Serviços externos ainda exigem conectividade estável e chaves válidas.
- **Persistência e espaço em disco:** logs, uploads e resumos ficam em `./data/api`. O volume cresce com novos experimentos; faça limpeza periódica ou mova os arquivos gerados para armazenamento frio. O índice do Weaviate consome `./data/weaviate` e pode ultrapassar centenas de MB dependendo da base.

## Estrutura de Arquivos

````
.
├── api/                # Backend FastAPI e lógica RAG (uploads, endpoints, vetores)
├── web/                # Frontend Streamlit com painel experimental
├── simulator/          # Publicador MQTT de telemetria e injeção de falhas
├── data/
│   ├── api/            # Uploads, índices FAISS, logs e resumos persistidos
│   └── weaviate/       # Volume local do contêiner Weaviate
├── docs/               # Papel, apresentação, gabarito oficial e instruções extras
├── notebooks/          # Consolidação opcional dos experimentos (Plotly/Pandas)
├── docker-compose.yml  # Orquestração
└── README.md
````

## Tecnologias Utilizadas

* **LLMs**: Groq Llama 3.x (8B/70B), Gemini 1.5/2.5 Flash, Ollama Llama3.2 3B para o modo offline.
* **RAG / Embeddings**: Sentence-Transformers (padrão `all-MiniLM-L6-v2`, com suporte a all-mpnet-base-v2, multi-qa-mpnet, gte-large, etc.), vetorização servida por ChromaDB, FAISS, Weaviate e Pinecone.
* **Backend/API**: FastAPI + Python, langchain-community para integração com vetores, pandas/plotly para sumarização experimental.
* **Frontend**: Streamlit (dashboard, upload, logging e controle de falhas) + Requests para chamadas à API.
* **IoT / Mensageria**: MQTT (paho-mqtt), broker Mosquitto público/local, comandos de falha em tempo real.


**Autores**: Amauri Cunha, Yessica Maria Valencia Lemos
**Data**: 06 de Dezembro 2025