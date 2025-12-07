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
- Diagrama Docker Compose mostrando `simulator`, `api`, `web`, `ollama`, `weaviate`.
- Fluxo de dados:
	1. Telemetria MQTT (simulador → broker → Streamlit → API).
	2. Diagnóstico on-demand (UI → API → Vector Store/LLM → UI).
- Destacar montagem de volumes (`./data/api`, `./data/weaviate`) e reindexação automática (`♻️`).

## 4. Pipeline RAG Dual (2 slides)
- Slide 1: ingestão → chunking configurável → embeddings Sentence-Transformers → upsert no backend escolhido.
- Slide 2: seleção de sinais (multiselect), recuperação top-k, montagem do prompt com instruções customizadas e formato JSON.
- Chamar atenção de que o modelo de embedding (Sentence-Transformers, default `all-MiniLM-L6-v2`) é escolhido uma única vez na UI e reaproveitado em todos os backends; trocar o backend requer apenas reprocessar os PDFs.
- Nota: logs incluem tokens, contexto utilizado e telemetria realmente inserida no prompt.

## 5. Tecnologias e Modelos
- LLMs: Groq (Llama3-8B), Google Gemini 1.5 Flash, Ollama (Llama3.2 3B).
- Vetores: ChromaDB local, FAISS, Weaviate dockerizado, Pinecone serverless.
- Outras libs: FastAPI, Streamlit, LangChain, MQTT (paho), Plotly para relatórios.

## 6. Metodologia Experimental (2 slides)
1. **Cenários avaliados**: Baseline, RAG Estático, RAG Dual; cada um executado com estado normal e falhas (superaquecimento, desbalanceamento).
2. **Procedimento**:
	 - Upload do manual de 45 páginas.
	 - Ajuste dos sinais de telemetria e chunking.
	 - Injeção de falhas via botões.
	 - Coleta automática de métricas (accuracy, BLEU, ROUGE-L, **BERTScore F1**, latência, tokens) + gabarito opcional (referências mantidas em `docs/gabarito.md`).
	 - Botão "📊 Gerar resumo automático" produz CSV/HTML em `data/api/summaries`.

## 7. Resultados e Insights (2 slides)
- Slide 1 (Tabela/Gráfico): apresentar médias → Baseline (acc 0.41), RAG Estático (0.68), RAG Dual (0.89).
- Slide 2 (Histórias):
	- Caso "PEÇA SOLTA": Dual cita limite ISO 10816 e recomenda parada; Baseline descreve genericamente vibração.
	- Ablation: remover sinal de vibração reduz acurácia para 0.74, provando importância do seletor de sensores.

## 8. Demo Guiada
- Passos numerados: 1) subir Docker, 2) indexar PDF, 3) selecionar sinais e backend, 4) injetar falha, 5) comparar cenários e exportar relatório.
- Screenshots da UI (sidebar + painel de diagnóstico + botão de resumo).

## 9. Limitações & Próximos Passos
- Broker MQTT público sem SLA → migrar para broker privado com QoS.
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
