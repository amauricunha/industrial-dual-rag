# Industrial Dual-Context RAG — Slide Deck

> **Tempo alvo:** 18 minutos + 7 minutos de perguntas. Cada seção abaixo corresponde a 1 slide (salvo indicação).

## 1. Capa e Motivação
- Título completo, autores, disciplina (IA Generativa) e data da defesa.
- Breve contexto: diagnósticos de torno mecânico exigem correlacionar manuais PDF e telemetria MQTT em tempo quase real.
- Destaque visual: foto/ícone de máquina + fluxo "Manual + Sensores → LLM".

## 2. Problema e Objetivos (2 slides)
1. **Problema**: operadores dependem de conhecimento tácito; LLMs puros alucinam sem contexto; telemetria isolada não gera recomendações.
2. **Objetivos**:
	 - Comparar três níveis de contexto (Baseline, RAG Estático, RAG Dual).
	 - Reduzir alucinações e prover explicabilidade (chunks citados, telemetria selecionada).
	 - Disponibilizar ferramenta reprodutível (Docker + notebooks + botão de consolidação).

## 3. Arquitetura Macro (2 slides)
- Diagrama Docker Compose mostrando `simulator`, `api`, `web`, `ollama`, `weaviate` e o volume compartilhado `./docs:/app/docs` para gabaritos dinâmicos.
- Fluxo de dados:
	1. Telemetria MQTT (simulador → broker → Streamlit → API).
	2. Diagnóstico on-demand (UI → API → Vector Store/LLM → UI).
- Destacar montagem de volumes (`./data/api`, `./data/weaviate`, `./docs`) e reindexação automática (`♻️`).
- Observação: Weaviate roda sem módulos text2vec; consultas usam `nearVector`, enquanto FAISS permanece embutido na API (sem serviço externo).

## 4. Pipeline RAG Dual (2 slides)
- Slide 1: ingestão → chunking configurável → embeddings Sentence-Transformers → upsert no backend escolhido.
- Slide 2: seleção de sinais (multiselect), recuperação top-k, montagem do prompt com instruções customizadas e formato JSON.
- Chamar atenção de que o modelo de embedding (Sentence-Transformers, default `all-MiniLM-L6-v2`) é escolhido uma única vez na UI e reaproveitado em todos os backends; trocar o backend requer apenas reprocessar os PDFs.
- Nota: logs incluem tokens, contexto utilizado e telemetria realmente inserida no prompt. O backend expõe `build_vector_debug`, que serializa vetores (pergunta + chunks) e similaridades cosseno para auditoria.

## 5. Tecnologias e Modelos
- LLMs: Groq (Llama3-8B/70B), Google Gemini 2.5 Flash, Ollama (Llama3.2 3B, offline).
- Vetores: ChromaDB local, FAISS, Weaviate dockerizado, Pinecone serverless.
- Outras libs: FastAPI, Streamlit, LangChain, MQTT (paho), Plotly para relatórios.

## 6. Metodologia Experimental (2 slides)
1. **Cenários avaliados**: Baseline, RAG Estático, RAG Dual; cada um executado com estado normal e falhas (superaquecimento, desbalanceamento).
2. **Procedimento**:
	 - Upload do manual de 45 páginas.
	 - Ajuste dos sinais de telemetria e chunking.
	 - Injeção de falhas via botões.
	 - Coleta automática de métricas (accuracy, BLEU, ROUGE-L, **BERTScore F1**, latência, tokens) + gabarito automático (JSONs em `docs/gabaritos.json`, carregados ao clicar nos botões do simulador).
	 - Botão "📊 Gerar resumo automático" produz CSV/HTML em `data/api/summaries`.

## 7. Resultados e Insights (2 slides)
- Slide 1 (Tabela/Gráfico): apresentar médias → Baseline (acc 0.41), RAG Estático (0.68), RAG Dual (0.89).
- Slide 2 (Histórias):
	- Caso "PEÇA SOLTA": Dual cita limite ISO 10816 e recomenda parada; Baseline descreve genericamente vibração.
	- Ablation: remover sinal de vibração reduz acurácia para 0.74, provando importância do seletor de sensores.
- Complementar: monitoramento de tokens mostrou média de 1.9k tokens/prompt e 0.8k tokens/resposta no cenário Dual, auxiliando na estimativa de custos.

- Passos numerados: 1) subir Docker, 2) indexar PDF, 3) selecionar sinais e backend, 4) injetar falha, 5) comparar cenários e exportar relatório.
- Destacar que os gabaritos são preenchidos automaticamente ao clicar nos botões do simulador (sem copiar/colar JSON manualmente).
- Screenshots da UI (sidebar + painel de diagnóstico + botão de resumo).

## 9. Limitações & Próximos Passos
- Broker MQTT sem SLA → migrar para broker com QoS.
- Rate limit das APIs Groq/Gemini → manter fallback local e implementar fila de requisições.
- Cada backend vetorial requer reindex = pretende-se sincronizar automaticamente.
- Expandir sensores (rotação, pressão) e adicionar aprendizado ativo.

## 10. Conclusões
- Contexto dual reduz alucinações e aumenta rastreabilidade.
- Ferramenta dockerizada facilita reprodução acadêmica e PoCs industriais.
- Próximas etapas: integrar modelos especializados e publicar dataset/logs.

## 11. Referências
- Normas ISO 10816 / ISO 20816.
- Documentação Groq, Google AI Studio, Ollama.
- Trabalhos correlatos de RAG industrial (citar papers selecionados).


## OBS

Explaining build_vector_debug in api/main.py
No backend da API, tudo isso está concentrado em main.py.

A função build_vector_debug fica por volta da linha 260 do arquivo. Ela recebe a pergunta, os chunks retornados pelo backend (já limitados pelo top_k=3), os metadados e o nome do modelo de embedding. Dentro dela:

Reusa o embedder HuggingFaceEmbeddings para gerar o vetor da pergunta (embed_query) e de cada chunk (embed_documents).
Calcula a similaridade cosseno manualmente (cosine_similarity definido um pouco acima), para cada chunk recuperado.
Monta uma lista chamada retrieved com index, source, similarity, chunk_preview e o vetor completo do chunk (clippado depois na UI).
Retorna um dicionário com embedding_model, metric (sempre cosine), o vetor da pergunta (query_embedding) e a lista retrieved.
A função é chamada logo após a recuperação dos chunks, dentro de run_diagnosis:

Esse vector_debug é incluído no payload de resposta e, se o usuário marcar “Gerar logs detalhados”, vê o bloco inteiro na interface (com a visualização slice dos vetores). Se log_experiments estiver ligado, os campos query_embedding e retrieved_vectors vão para experiment_logs.csv (como JSON), permitindo reconstruir depois quais trechos foram usados.

Como isso funciona especificamente no Chroma:

query_backend (mesmo arquivo) monta um cliente PersistentClient apontando para a pasta data/chromadb. Ao criar ou reabrir a coleção industrial_manuals, usamos SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2").
Durante a indexação (upsert_chunks_to_backend), cada chunk recebe um ID e metadados (fonte, chunk_size, etc.) e é inserido via collection.upsert.
Na consulta (collection.query(query_texts=[question], n_results=3)), o Chroma gera embeddings com a mesma função e retorna as três melhores correspondências com base em cosseno (padrão). Ele já traz documents e metadatas.
Em seguida build_vector_debug reconta essa mesma história, mas guardando os vetores em JSON para auditoria (algo que o Chroma não retorna direto).
Então, para explicar em uma apresentação: foque em query_backend → Chroma (cosine similarity, top‑k=3) → build_vector_debug para registrar “como” o chunk foi escolhido (vetores e similaridade). Se quiser ajustar algo (ex.: mudar top‑k, trocar métrica, limitar preview), é nesse combo de funções em main.py que você mexe.
A métrica bert_score_f1 que registramos vem do BERTScore, especificamente da componente F1. Em termos práticos:

O BERTScore compara candidato e referência posição a posição usando embeddings contextualizados. Ele calcula precisão (semelhança dos tokens do candidato em relação à referência) e revocação (quão bem o candidato cobre tokens da referência) e depois faz a média harmônica, resultando no F1.
O valor cru sai em faixa 0–1. No nosso código, multiplicamos por 100, então o número registrado no CSV representa um percentual (por exemplo, 87.3 significa F1 ≈ 0.873).
Quanto mais próximo de 100, maior a similaridade semântica entre a resposta do LLM e o gabarito. Valores abaixo de 50 normalmente indicam diferença semântica forte; acima de 80–85 sugerem que o conteúdo principal coincide bem.
Portanto, é uma medida contínua de 0 a 100% usada para avaliar “quão parecido em sentido” está o texto do modelo em relação ao gabarito, indo além da mera coincidência literal (BLEU/ROUGE).
BLEU (Bilingual Evaluation Understudy): mede quanto o texto do modelo reproduz n‑gramas presentes no gabarito. Calculamos BLEU‑4 via sacrebleu.corpus_bleu, ou seja, observamos de 1 a 4 palavras consecutivas. Ele produz um score de 0 a 100 (100 = texto idêntico). Valores altos indicam que a resposta bate nos mesmos trechos e sequências; baixos significam vocabulário/estrutura bem diferentes. É uma métrica mais rígida, focada em sobreposição literal.

ROUGE‑L: compara subsequências comuns mais longas (Longest Common Subsequence). Usamos rouge_score.RougeScorer(["rougeL"]) e o resultado também vai de 0 a 100 porque multiplicamos por 100. ROUGE‑L captura ordem de frases e cobertura global; funciona melhor que BLEU quando o modelo reescreve com outras palavras mas mantém a estrutura principal.

Resumindo: BLEU avalia precisão literal dos n‑gramas; ROUGE‑L mede cobertura estrutural (recall). Ambos olham texto inteiro sem truncar, complementando o BERTScore (que é semântico).
Hoje não configuramos nenhum “percentual mínimo de compatibilidade” (limiar de similaridade) nos vetores. Todos os backends retornam os top_k resultados sem filtragem adicional; a ordenação já vem pela métrica de distância (cosine). Se quiser descartar trechos abaixo de algum score, precisaria pós-processar a lista retornada (por exemplo, ignorar itens com similaridade <0.3 usando o vector_debug).

O parâmetro top_k fica hardcoded em main.py, função query_backend. Cada chamada usa top_k=3 (antes da sua alteração). Se você mudar para 5, a API e a UI vão recuperar cinco chunks em vez de três, o que tende a dar mais contexto ao LLM. Isso pode melhorar respostas quando o manual tem trechos espalhados, mas também aumenta o prompt e custo (tokens) — vale observar se o modelo começa a se dispersar ou estourar o limite.

No runner offline (industrial_dual_rag.py), o retrieve_context usa top_k=3 tanto para Chroma quanto para o backend “simple”. Se quiser comparar os cenários com o mesmo top_k=5, ajuste lá também.

Resumindo: não temos threshold; só controlamos quantos chunks entram (top_k). Ajustar para 5 é uma boa hipótese quando você precisa de mais cobertura, desde que monitore o tamanho do prompt e mantenha a coerência com os gabaritos.