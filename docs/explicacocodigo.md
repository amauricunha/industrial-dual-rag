# Explicação do Código

Este documento resume como os principais componentes da aplicação (API FastAPI e painel Streamlit) funcionam por dentro. Use-o como referência rápida ao revisar o repositório `industrial-dual-rag`.

## 1. API (`api/main.py`)

### 1.1 Estrutura geral
- Carrega variáveis de ambiente e define caminhos persistentes em `DATA_DIR`, mantendo uploads, índices e logs dentro do volume `./data/api`.
- Inicializa um `FastAPI` com endpoints para upload/reindexação de PDFs, execução de diagnósticos (`/chat`), logging de experimentos e consolidação de métricas.
- Mantém um cliente `chromadb.PersistentClient` compartilhado e caches de embeddings `HuggingFaceEmbeddings` para evitar recarregamentos caros.

### 1.2 Ingestão de documentos
- `upload_manual` recebe um PDF via `UploadFile`, grava em `UPLOAD_DIR` e chama:
  - `extract_text_from_pdf` → usa `pypdf` para juntar o texto das páginas.
  - `chunk_text` → divide em janelas configuráveis (`chunk_size`, `chunk_overlap`).
  - `upsert_chunks_to_backend` → envia os chunks para o backend vetorial escolhido (`chroma`, `faiss`, `weaviate`, `pinecone`). Cada metadado inclui `source`, `chunk_size`, `embedding_model` e `backend`.
- `reindex_manuals` reutiliza PDFs já salvos para regenerar embeddings quando o usuário altera backend ou parâmetros de chunking.

### 1.3 Consulta vetorial e debug
- `query_backend` aplica busca semântica com `top_k=3` em todos os backends. No Chroma usamos `collection.query(..., n_results=top_k)`, nos demais LangChain faz `similarity_search(..., k=top_k)`.
- `build_vector_debug` reconstrói os vetores da pergunta e dos chunks recuperados usando o mesmo `HuggingFaceEmbeddings`, computa similaridade cosseno (`cosine_similarity`) e envia previews + embeddings completos no payload de resposta e no CSV (quando logging está ativo).

### 1.4 Telemetria e montagem do prompt
- `build_telemetry_section` normaliza o snapshot enviado pela UI, aplica as chaves selecionadas e gera o bloco textual com alertas (“Temperatura acima do limite crítico”). Também retorna o dicionário filtrado para logging.
- `run_diagnosis` é o endpoint central:
  1. Determina cenário (baseline, RAG docs, RAG + telemetria) e recupera chunks conforme necessário.
  2. Monta seções opcionais de instruções e formato JSON (`response_format`).
  3. Concatena `base_system`, contexto estático, telemetria e pergunta no `final_prompt`.
  4. Chama `get_llm_response`, que abstrai Groq, Gemini ou Ollama (cada um com seu SDK). Não há fallback simulado—se a chamada real falhar, retornamos o erro.
  5. Estima tokens com `estimate_tokens` (tiktoken se disponível, caso contrário contagem de palavras) e devolve metadados (modo usado, backend vetorial, vetores, tokens, sinais de telemetria aplicados).

### 1.5 Métricas, logging e relatórios
- `compute_text_metrics` não vive na API; a UI calcula accuracy/BLEU/ROUGE/BERTScore localmente. A API apenas recebe os valores via `/experiments/log` e persiste no CSV `experiment_logs.csv`.
- `ensure_experiment_log_schema` garante que o CSV tenha o cabeçalho esperado, reescrevendo linhas existentes quando evoluímos as colunas.
- `generate_experiment_summary` lê o CSV, agrega métricas (por cenário e modo), exporta `summary_metrics.csv`, `recent_samples.csv` e gráficos Plotly (HTML). Também limpa artefatos antigos para evitar confusão.

## 2. Painel Streamlit (`web/app.py`)

### 2.1 Configuração e estado
- Carrega `.env`, define constantes (modelos default, backends suportados, sinais de telemetria) e chama `st.set_page_config`.
- Usa `st.session_state` para persistir telemetria, histórico de diagnósticos, caches de modelos LLM e parâmetros de chunking/embedding.
- Implementa cache de métricas semânticas com `_instantiate_bert_scorer` + `get_bert_scorer`. Agora o app carrega também o tokenizer (`AutoTokenizer`) para truncar entradas longas via `truncate_for_bertscore`, mantendo a métrica BERTScore estável.

### 2.2 MQTT e simulador
- `start_mqtt` configura o cliente `paho.mqtt.client`, assina o tópico definido pelas variáveis `MQTT_BROKER`, `MQTT_TOPIC_SENSORS` e deposita mensagens numa `Queue` compartilhada (`get_mqtt_queue`).
- `pump_mqtt_queue` atualiza `st.session_state.telemetry`, alimentando os cards do painel e o payload enviado ao backend.
- Botões “Operação Normal”, “Falha Térmica”, “Desbalanceamento” chamam `publish_command`, que publica comandos MQTT para o simulador.

### 2.3 UI e interação com a API
- Sidebar controla provedor/modelo LLM, parâmetros de chunking, backend vetorial e seleção de sinais. Uploads/reprocessamentos fazem POST para `/upload` e `/reindex`.
- A área principal mostra telemetria (cards), controles do simulador, seleção de cenário (1, 2, 3) e editor do prompt (base system, instruções, JSON de saída).
- O botão “Gerar Relatório de Diagnóstico” monta o payload e chama `POST /chat`. Quando o checkbox “Gravar logs de experimentos” está ativo, a UI:
  1. Solicita o gabarito (referência) do usuário.
  2. Executa `compute_text_metrics`, calculando accuracy, BLEU, ROUGE-L e BERTScore (com truncamento/tokenizer alinhado ao core).
  3. Chama `persist_experiment_log`, enviando métricas + vetores para a API.
- O botão “📊 Gerar resumo automático” dispara `/experiments/summarize` e exibe os artefatos gerados na pasta `data/api/summaries`.

### 2.4 Visualização de resultados
- Após uma chamada bem-sucedida, o app exibe o relatório retornado pela API, destacando o modo utilizado, backend vetorial, uso de tokens e se houve contexto.
- Se `debug` estiver marcado, mostramos:
  - Prompt final concatenado.
  - Chunks recuperados e seus metadados.
  - Telemetria de fato inserida no prompt.
  - `vector_debug` com pré-visualização dos embeddings e similaridade (primeiras dimensões em tabela + expander com vetor completo).

## 3. Relação entre API e Web

1. **Ingestão**: UI chama `/upload`/`/reindex` para manter os vetores sincronizados. Não há lógica de chunking no Streamlit—tudo reside na API.
2. **Diagnóstico**: UI monta o payload com a telemetria em tempo real, instruções e formato de resposta; API combina com os chunks recuperados e chama o LLM selecionado.
3. **Métricas**: UI calcula metrics e envia para `/experiments/log`. O core offline (`core/industrial_dual_rag.py`) usa a mesma lógica (inclusive truncamento de BERTScore) para garantir consistência.
4. **Auditoria**: `vector_debug`, tokens e telemetria são retornados pela API e apresentados na UI/logs, permitindo explicar “como” o diagnóstico foi produzido.

Com este panorama você consegue navegar pelo código e entender onde cada parte da lógica reside sem precisar reler todos os arquivos do zero.