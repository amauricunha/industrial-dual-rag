# Copilot Instructions

## System Snapshot
- Three services live in `docker-compose.yml`: FastAPI backend (`api/`), Streamlit operator console (`web/`), and MQTT simulator (`simulator/`). Ollama and Weaviate run as sidecars when local LLM or external vector DB are needed. All persistent artifacts mount under `./data` (keep uploads/logs between runs).
- Boot locally with `docker compose up --build -d`. The Streamlit UI at `:8501` is the main manual test harness; FastAPI listens on `:8000`. Use `bash clean_data.sh --yes` when you need to wipe every index/upload before a new experiment cycle.

## Backend (FastAPI) Patterns
- `api/main.py` is single-file but organized por "Etapa" banners. Keep related helpers near their stage so the pipeline narrative stays intact.
- Ingest flow (`upload_manual`, `extract_text_from_pdf`, `chunk_text`, `upsert_chunks_to_backend`) must always populate the metadata quartet `{source, chunk_size, chunk_overlap, embedding_model}` plus `backend`. These dicts are stored inside every vector store and power `vector_debug` and CSV logs.
- Retrieval lives in `query_backend`: always return `(docs, metas)` (text + metadata) and keep `top_k=3` aligned with the docs. If you add a backend, mirror the existing implementations (Chroma native, others via LangChain) and ensure `ensure_backend_supported` lists it.
- Diagnostics assemble prompts in `run_diagnosis`: scenario 2 pulls static chunks, scenario 3 also calls `build_telemetry_section`. Any new context block must be added before the final `=== SOLICITAÇÃO DO OPERADOR ===` marker so debug views stay coherent.
- Logging: `/experiments/log` writes rows using `EXPERIMENT_LOG_COLUMNS`. When extending telemetry or metric fields, update `EXPERIMENT_LOG_COLUMNS`, `ensure_experiment_log_schema`, and `generate_experiment_summary` together.
- `generate_experiment_summary` always rewrites HTML charts inside `SUMMARY_OUTPUT_DIR`. If you add metrics, create the chart builder there and remember to drop stale HTML files before saving.

## Frontend (Streamlit) Patterns
- `web/app.py` mirrors the backend stages (comments `Etapa Frontend`). Reuse `st.session_state` keys defined near the top; new persistent knobs should be initialized in the same block to survive reruns.
- Metric computation happens client-side: `compute_text_metrics` calls BLEU/ROUGE + BERTScore. If you change BERTScore behavior, adjust `_instantiate_bert_scorer`, `get_active_bert_tokenizer`, and `truncate_for_bertscore` together.
- MQTT handling (`start_mqtt`, `pump_mqtt_queue`, `publish_command`) already abstracts queueing. Stick to these helpers instead of directly touching `mqtt_queue` elsewhere.
- Any payload sent to `/chat` is built inside the "Gerar Relatório" button handler (`web/app.py#L907-L1058`). Keep it the single source of truth for request fields so logging and prompt assembly stay in sync with the backend models.
- Debugging UI relies on `vector_debug` plus `render_vector_preview`. If backend debug payload changes, update this expander to avoid mismatches.

## Simulator & Telemetry
- `simulator/main.py` publishes four signals (status, temperature, vibration, current) every 2 seconds and reacts to commands `NORMAL`, `HIGH_TEMP`, `HIGH_VIBRATION`. Keep any new variables synchronized with `TELEMETRY_SIGNAL_OPTIONS` in the web app and with `build_telemetry_section` in the API.

## Developer Workflows
- Reindexing: when changing `vector_backend`, chunk params, or `embedding_model`, either hit the UI button `♻️ Reprocessar base existente` or call `/reindex`. The function reads PDFs from `/app/data/uploads`, so never delete that folder unless you intend to lose the source documents.
- Vector resets: Chroma lives under `data/api/chromadb`, FAISS under `data/api/faiss_index`, Weaviate under `data/weaviate`. Deleting one store requires reindexing only that backend.
- Logging & reports: enable "Gravar logs de experimentos" in the UI to populate `experiment_logs.csv`. Use `/experiments/summarize` (triggered by the sidebar button) to regenerate Plotly dashboards in `SUMMARY_OUTPUT_DIR`.
- LLM keys: `.env` is shared by all services. Groq/Gemini keys unlock external inference; Ollama is preloaded with `${OLLAMA_BOOT_MODEL}`. Update `.env` rather than hardcoding secrets.

## Conventions & Gotchas
- Stick to Portuguese labels/messages—the UI, docs, and README follow PT-BR, and automated summaries expect that tone.
- Scenarios are fixed (`1: Zero-Shot`, `2: RAG Estático`, `3: RAG Dual`). Any new mode must update `SCENARIO_LABELS`, `run_diagnosis` mode map, and the Streamlit radio options together.
- `telemetry_signals` flows from the UI multiselect → API request → `build_telemetry_section`. Keep the allowed keys consistent across these touch points.
- There are no automated tests; validation is done interactively through the dashboard and by inspecting `data/api/experiment_logs.csv`. When making risky changes, spin up Docker, inject simulator faults, and verify summaries regenerate without errors.

Feedback welcome—ping if any workflow above is unclear or if you need more detail on a specific component.
