# Explicação do Código

Este documento resume como os principais componentes da aplicação (API FastAPI e painel Streamlit) funcionam por dentro. Use-o como referência rápida ao revisar o repositório `industrial-dual-rag`.

## 1. API (`api/main.py`)

### 1.1 Estrutura geral
- [`Configuração global`](../api/main.py#L1-L117) carrega variáveis de ambiente, inicializa caminhos (`DATA_DIR`, `UPLOAD_DIR`, `SUMMARY_OUTPUT_DIR`) e garante a criação das pastas dentro do volume `./data/api`.
- [`FastAPI` + modelos Pydantic](../api/main.py#L996-L1450) definem endpoints para upload/reindexação, `/chat`, logging e consolidação de métricas.
- [`Clientes vetoriais e cache de embeddings`](../api/main.py#L90-L170) mantêm um `chromadb.PersistentClient` compartilhado e instâncias `HuggingFaceEmbeddings` reutilizáveis para evitar recarregamentos caros.

### 1.2 Ingestão de documentos
- [`upload_manual`](../api/main.py#L1004-L1071) recebe um PDF via `UploadFile`, grava no `UPLOAD_DIR`, valida parâmetros e dispara as etapas abaixo:
  - [`extract_text_from_pdf`](../api/main.py#L212-L235) usa `pypdf` para juntar o texto das páginas, com tratamento de erro detalhado.
  - [`chunk_text`](../api/main.py#L177-L195) divide o texto em janelas configuráveis (`chunk_size`, `chunk_overlap`).
  - [`upsert_chunks_to_backend`](../api/main.py#L540-L620) distribui os chunks para o backend escolhido (`chroma`, `faiss`, `weaviate`, `pinecone`), preenchendo metadados (`source`, `chunk_size`, `embedding_model`, `backend`).
- [`reindex_manuals`](../api/main.py#L1072-L1161) reaproveita PDFs já armazenados para regenerar embeddings ao trocar backend ou parâmetros de chunking, contabilizando arquivos processados e ignorados.

### 1.3 Consulta vetorial e debug
- [`query_backend`](../api/main.py#L622-L697) executa busca semântica com `top_k=3` para todos os backends: usa `collection.query(..., n_results=top_k)` no Chroma e `similarity_search(..., k=top_k)` nas integrações LangChain.
- [`build_vector_debug`](../api/main.py#L735-L784) reconstrói embeddings da pergunta e dos chunks usando o mesmo `HuggingFaceEmbeddings`, calcula similaridade cosseno com [`cosine_similarity`](../api/main.py#L722-L733) e inclui previews/vetores completos no payload e no CSV quando logging está ativo.

### 1.4 Telemetria e montagem do prompt
- [`build_telemetry_section`](../api/main.py#L786-L828) normaliza o snapshot da UI, respeita `telemetry_signals`, gera alertas e devolve o dicionário filtrado para logging.
- [`run_diagnosis`](../api/main.py#L1301-L1447) orquestra todo o fluxo:
  1. Decide o cenário e, se necessário, chama novamente [`query_backend`](../api/main.py#L622-L697) para obter contexto estático.
  2. Monta blocos opcionais de instruções (`instructions_block`) e formato JSON (`response_format_block`).
  3. Concatena `base_system`, contexto estático, telemetria formatada e pergunta no `final_prompt`.
  4. Invoca [`get_llm_response`](../api/main.py#L850-L991), que encapsula Groq, Gemini e Ollama (sem mocks: qualquer erro real é propagado).
  5. Chama [`estimate_tokens`](../api/main.py#L698-L711) para estimar o uso de tokens e devolve metadados (modo utilizado, backend, vetores, telemetria aplicada).

### 1.5 Métricas, logging e relatórios
- A UI calcula métricas via [`compute_text_metrics`](../web/app.py#L497-L533); a API apenas recebe os valores em [`log_experiment`](../api/main.py#L1245-L1286) e persiste no CSV `experiment_logs.csv`.
- [`ensure_experiment_log_schema`](../api/main.py#L488-L531) garante que o CSV tenha o cabeçalho atualizado sempre que novas colunas são introduzidas.
- [`generate_experiment_summary`](../api/main.py#L236-L487) consolida o histórico, salva `summary_metrics.csv`, `recent_samples.csv` e gráficos Plotly (HTML), além de limpar artefatos antigos para evitar confusões.

## 2. Painel Streamlit (`web/app.py`)

### 2.1 Configuração e estado
- [`env_or_default` + constantes iniciais](../web/app.py#L38-L167) carregam `.env`, definem modelos/backends padrão e finalizam com `st.set_page_config`.
- O bloco de [`st.session_state`](../web/app.py#L286-L377) persiste telemetria, histórico de diagnósticos, caches de modelos e parâmetros de chunking/embedding.
- Métricas semânticas são tratadas por [`get_bert_scorer`](../web/app.py#L233-L283), [`get_active_bert_tokenizer`](../web/app.py#L181-L190) e [`truncate_for_bertscore`](../web/app.py#L192-L211), mantendo tokens alinhados ao cálculo de BERTScore.

### 2.2 MQTT e simulador
- [`start_mqtt`](../web/app.py#L349-L381) configura o cliente `paho.mqtt.client`, assina `MQTT_TOPIC_SENSORS` e injeta mensagens na [`Queue`](../web/app.py#L319-L333) criada por `get_mqtt_queue`.
- [`pump_mqtt_queue`](../web/app.py#L384-L402) atualiza `st.session_state.telemetry`, abastecendo os cards e o payload enviado aos endpoints.
- Botões “Operação Normal/Falha Térmica/Desbalanceamento” chamam [`publish_command`](../web/app.py#L439-L454), que publica comandos MQTT para o simulador.

### 2.3 UI e interação com a API
- A [`sidebar`](../web/app.py#L584-L777) controla provedor/modelo LLM, parâmetros de chunking, backend vetorial, seleção de sinais e upload/reprocessamento (POST `/upload` e `/reindex`).
- A área principal entre [`st.title` e o painel de controle`](../web/app.py#L778-L905) mostra telemetria, simulador de falhas e seleção de cenário (1–3) com configuração de prompts.
- O botão [“Gerar Relatório de Diagnóstico”](../web/app.py#L907-L1058) monta o payload completo e chama `POST /chat`. Quando “Gravar logs de experimentos” está ativo:
  1. Solicita o gabarito ao usuário.
  2. Executa [`compute_text_metrics`](../web/app.py#L497-L533) para accuracy/BLEU/ROUGE-L/BERTScore.
  3. Envia tudo via [`persist_experiment_log`](../web/app.py#L456-L495), incluindo `vector_debug` quando disponível.
- O botão [“📊 Gerar resumo automático”](../web/app.py#L738-L777) dispara `/experiments/summarize` e exibe os artefatos da pasta `data/api/summaries`.

### 2.4 Visualização de resultados
- Após uma resposta, o painel em [`st.session_state.diagnosis_history`](../web/app.py#L1061-L1235) destaca modo utilizado, backend vetorial, tokens e contexto.
- Com `debug` ativo, mostramos o prompt final, chunks e telemetria. O bloco usa [`render_vector_preview`](../web/app.py#L535-L567) para exibir `vector_debug`, incluindo expansor com o vetor completo.

## 3. Relação entre API e Web

1. **Ingestão**: a UI chama [`/upload`](../api/main.py#L1004-L1071) e [`/reindex`](../api/main.py#L1072-L1161) para manter os vetores sincronizados; todo o chunking/embedding vive na API.
2. **Diagnóstico**: o front monta o payload no [handler do botão de diagnóstico](../web/app.py#L907-L1058), enquanto a API consolida tudo em [`run_diagnosis`](../api/main.py#L1301-L1447) antes de chamar o LLM escolhido.
3. **Métricas**: o Streamlit calcula métricas via [`compute_text_metrics`](../web/app.py#L497-L533) e envia para [`/experiments/log`](../api/main.py#L1245-L1286); a API apenas persiste e consolida (via [`generate_experiment_summary`](../api/main.py#L236-L487)).
4. **Auditoria**: `vector_debug`, tokens e telemetria retornam de [`run_diagnosis`](../api/main.py#L1301-L1447) e são apresentados pela UI (debug expander + [`render_vector_preview`](../web/app.py#L535-L567)) para explicar “como” o diagnóstico foi produzido.

Com este panorama você consegue navegar pelo código e entender onde cada parte da lógica reside sem precisar reler todos os arquivos do zero.