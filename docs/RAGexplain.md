# Pipeline RAG Industrial Dual

Este documento resume o fluxo ponta a ponta do RAG dual da API FastAPI, destacando onde cada etapa ocorre no código de `api/main.py`. Para cada fase há um comentário curto seguido do trecho original solicitado.

## 1. Ingestão de PDFs, chunking e embeddings
`upload_manual` faz o upload, chama `extract_text_from_pdf`, divide o texto com `chunk_text` e gera metadados antes de indexar via `upsert_chunks_to_backend`.

```python
@app.post("/upload")
async def upload_manual(
    file: UploadFile = File(...),
    vector_backend: Optional[str] = Form(None),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Apenas PDFs.")

    backend_choice = (vector_backend or DEFAULT_VECTOR_BACKEND).lower()
    ensure_backend_supported(backend_choice)

    if chunk_size <= 0:
        raise HTTPException(400, "chunk_size deve ser maior que zero.")
    if chunk_overlap < 0:
        raise HTTPException(400, "chunk_overlap não pode ser negativo.")
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Processamento simples de PDF
    text = extract_text_from_pdf(file_path)
    
    # Chunking/Indexação: parte central do requisito RAG (transforma o PDF em
    # pedaços com metadados prontos para qualquer backend vetorial escolhido).
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise HTTPException(400, "Documento vazio após processamento.")
    metadatas = [
        {
            "source": file.filename,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "backend": backend_choice,
            "embedding_model": embedding_model,
        }
        for _ in chunks
    ]

    try:
        upsert_chunks_to_backend(backend_choice, chunks, metadatas, embedding_model)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Falha ao indexar documento no backend %s", backend_choice)
        raise HTTPException(500, f"Erro ao indexar documento: {exc}") from exc
    
    # Atualiza lista
    current = list_files()
    if file.filename not in current:
        current.append(file.filename)
        with open(KB_Record_File, "w") as f: json.dump(current, f)
            
    return {
        "status": "indexed",
        "chunks": len(chunks),
        "backend": backend_choice,
        "embedding_model": embedding_model,
    }
```

O chunking em si garante sobreposição controlada e validação de parâmetros.

```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise HTTPException(400, "chunk_size deve ser maior que zero.")
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks
```

## 2. Indexação nos backends vetoriais
`upsert_chunks_to_backend` encapsula a gravação em Chroma, FAISS, Weaviate ou Pinecone reaproveitando os mesmos embeddings.

```python
def upsert_chunks_to_backend(
    backend: str,
    chunks: List[str],
    metadatas: List[dict],
    embedding_model: str,
):
    backend = backend.lower()
    ensure_backend_supported(backend)
    if backend == "chroma":
        ids = [f"{meta.get('source', 'manual')}:{idx}" for idx, meta in enumerate(metadatas)]
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        return

    if backend == "faiss":
        embedding = get_embedding_function(embedding_model)
        if os.path.isdir(FAISS_INDEX_DIR):
            store = FAISSVectorStore.load_local(
                FAISS_INDEX_DIR,
                embeddings=embedding,
                allow_dangerous_deserialization=True,
            )
            store.add_texts(chunks, metadatas=metadatas)
        else:
            store = FAISSVectorStore.from_texts(chunks, embedding=embedding, metadatas=metadatas)
        store.save_local(FAISS_INDEX_DIR)
        return

    if backend == "weaviate":
        if not WeaviateVectorStore or not weaviate:
            raise HTTPException(500, "Dependências do Weaviate não instaladas no backend.")
        url = os.getenv("WEAVIATE_URL")
        if not url:
            raise HTTPException(400, "Configure WEAVIATE_URL para usar este backend.")
        api_key = os.getenv("WEAVIATE_API_KEY")
        auth = weaviate.AuthApiKey(api_key) if api_key else None
        client = weaviate.Client(url=url, auth_client_secret=auth)
        index_name = os.getenv("WEAVIATE_CLASS", "IndustrialManual")
        embedding = get_embedding_function(embedding_model)
        store = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key="text",
            embedding=embedding,
            by_text=False,
        )
        store.add_texts(chunks, metadatas=metadatas)
        return

    if backend == "pinecone":
        if not PineconeVectorStore or not pinecone:
            raise HTTPException(500, "Dependências do Pinecone não instaladas no backend.")
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        index_name = os.getenv("PINECONE_INDEX", "industrial-dual-rag")
        if not api_key or not environment:
            raise HTTPException(400, "Configure PINECONE_API_KEY e PINECONE_ENVIRONMENT para usar este backend.")
        pc = pinecone.Pinecone(api_key=api_key)
        existing = {item["name"] for item in pc.list_indexes()}
        if index_name not in existing:
            dimension = int(os.getenv("PINECONE_DIMENSION", "384"))
            cloud = os.getenv("PINECONE_CLOUD", "aws")
            region = os.getenv("PINECONE_REGION", "us-east-1")
            if not ServerlessSpec:
                raise HTTPException(500, "pinecone-serverless não disponível.")
            pc.create_index(
                index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        index = pc.Index(index_name)
        embedding = get_embedding_function(embedding_model)
        namespace = os.getenv("PINECONE_NAMESPACE", "default")
        store = PineconeVectorStore(index=index, embedding=embedding, text_key="text", namespace=namespace)
        store.add_texts(chunks, metadatas=metadatas)
        return
```

## 3. Recuperação de contexto
`query_backend` consulta o mesmo backend configurado e devolve pares (texto, metadados) alinhados com `top_k=3`.

```python
def query_backend(
    backend: str,
    question: str,
    embedding_model: str,
    top_k: int = 3,
) -> tuple[List[str], List[dict]]:
    backend = backend.lower()
    ensure_backend_supported(backend)

    if backend == "chroma":
        results = collection.query(query_texts=[question], n_results=top_k)
        documents = results.get("documents") if results else None
        if documents and documents[0]:
            docs = documents[0]
            metas = (results.get("metadatas") or [[]])[0]
            return docs, metas
        return [], []

    if backend == "faiss":
        if not os.path.isdir(FAISS_INDEX_DIR):
            return [], []
        embedding = get_embedding_function(embedding_model)
        store = FAISSVectorStore.load_local(
            FAISS_INDEX_DIR,
            embeddings=embedding,
            allow_dangerous_deserialization=True,
        )
        docs = store.similarity_search(question, k=top_k)
        return [doc.page_content for doc in docs], [doc.metadata for doc in docs]

    if backend == "weaviate":
        if not WeaviateVectorStore or not weaviate:
            raise HTTPException(500, "Dependências do Weaviate não instaladas no backend.")
        url = os.getenv("WEAVIATE_URL")
        if not url:
            raise HTTPException(400, "Configure WEAVIATE_URL para consultar este backend.")
        api_key = os.getenv("WEAVIATE_API_KEY")
        auth = weaviate.AuthApiKey(api_key) if api_key else None
        client = weaviate.Client(url=url, auth_client_secret=auth)
        index_name = os.getenv("WEAVIATE_CLASS", "IndustrialManual")
        embedding = get_embedding_function(embedding_model)
        store = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key="text",
            embedding=embedding,
            by_text=False,
        )
        docs = store.similarity_search(question, k=top_k)
        return [doc.page_content for doc in docs], [doc.metadata for doc in docs]

    if backend == "pinecone":
        if not PineconeVectorStore or not pinecone:
            raise HTTPException(500, "Dependências do Pinecone não instaladas no backend.")
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        index_name = os.getenv("PINECONE_INDEX", "industrial-dual-rag")
        if not api_key or not environment:
            raise HTTPException(400, "Configure PINECONE_API_KEY e PINECONE_ENVIRONMENT para consultar este backend.")
        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        embedding = get_embedding_function(embedding_model)
        namespace = os.getenv("PINECONE_NAMESPACE", "default")
        store = PineconeVectorStore(index=index, embedding=embedding, text_key="text", namespace=namespace)
        docs = store.similarity_search(question, k=top_k)
        return [doc.page_content for doc in docs], [doc.metadata for doc in docs]

    return [], []
```

## 4. Telemetria dinâmica
`build_telemetry_section` normaliza o snapshot MQTT, aplica limites e devolve o bloco que será inserido no prompt.

```python
def build_telemetry_section(telemetry: Optional[dict], allowed_keys: Optional[List[str]] = None):
    if not telemetry:
        return "", {}

    normalized = {
        "status": telemetry.get("status", "N/A"),
        "temperature": safe_float(telemetry.get("temperature", 0)),
        "vibration": safe_float(telemetry.get("vibration", 0)),
        "current": safe_float(telemetry.get("current", 0)),
    }

    keys_to_use = normalize_telemetry_keys(allowed_keys)

    telemetry_lines = [
        "=== TELEMETRIA EM TEMPO REAL (CONTEXTO DINÂMICO) ===",
    ]

    filtered_snapshot = {}

    if "status" in keys_to_use:
        telemetry_lines.append(f"- Status Máquina: {normalized['status']}")
        filtered_snapshot["status"] = normalized["status"]
    if "temperature" in keys_to_use:
        telemetry_lines.append(f"- Temperatura: {normalized['temperature']:.1f} °C")
        filtered_snapshot["temperature"] = normalized["temperature"]
    if "vibration" in keys_to_use:
        telemetry_lines.append(f"- Vibração RMS: {normalized['vibration']:.2f} mm/s")
        filtered_snapshot["vibration"] = normalized["vibration"]
    if "current" in keys_to_use:
        telemetry_lines.append(f"- Corrente Motor: {normalized['current']:.1f} A")
        filtered_snapshot["current"] = normalized["current"]

    if "temperature" in keys_to_use and normalized["temperature"] > 90:
        telemetry_lines.append("ALERTA: Temperatura acima do limite crítico.")
    if "vibration" in keys_to_use and normalized["vibration"] > 10:
        telemetry_lines.append("ALERTA: Vibração excessiva detectada.")

    return "\n".join(telemetry_lines) + "\n", filtered_snapshot
```

## 5. Montagem do prompt e chamada ao LLM
`run_diagnosis` considera o cenário (baseline, RAG estático ou RAG dual), coleta chunks/telemetria, monta o prompt e chama `get_llm_response`. O payload retornado inclui metadados e telemetria para auditoria.

```python
@app.post("/chat")
def run_diagnosis(req: ChatRequest):
    base_system = (req.base_system or DEFAULT_BASE_SYSTEM).strip()
    vector_backend = (req.vector_backend or DEFAULT_VECTOR_BACKEND).lower()
    embedding_model = req.embedding_model or DEFAULT_EMBEDDING_MODEL
    ensure_backend_supported(vector_backend)
    telemetry_keys = normalize_telemetry_keys(req.telemetry_signals)
    
    context_part = ""
    telemetry_part = ""
    retrieved_chunks: List[str] = []
    retrieved_metadatas: List[dict] = []
    telemetry_snapshot = {}
    instructions_block = ""
    response_format_block = ""
    vector_debug: dict = {}

    if req.instructions:
        instructions_block = "\n=== INSTRUÇÕES OPERACIONAIS ===\n" + "\n".join(req.instructions) + "\n"
    if req.response_format:
        response_format_block = (
            "\n=== FORMATO DE RESPOSTA (JSON) ===\n"
            + json.dumps(req.response_format, indent=2, ensure_ascii=False)
            + "\n"
        )
    
    if req.scenario in [2, 3]:
        retrieved_chunks, retrieved_metadatas = query_backend(
            vector_backend,
            req.question,
            embedding_model,
            top_k=3,
        )
        if retrieved_chunks:
            doc_text = "\n---\n".join(retrieved_chunks)
            if doc_text.strip():
                context_part = f"\n=== BASE DE CONHECIMENTO (MANUAIS TÉCNICOS) ===\n{doc_text}\n"

    if req.scenario == 3:
        telemetry_part, telemetry_snapshot = build_telemetry_section(req.telemetry, telemetry_keys)

    vector_debug = build_vector_debug(req.question, retrieved_chunks, retrieved_metadatas, embedding_model)

    final_prompt = (
        f"{base_system}\n"
        f"{instructions_block}"
        f"{response_format_block}"
        f"{context_part}"
        f"{telemetry_part}"
        f"\n=== SOLICITAÇÃO DO OPERADOR ===\n{req.question}\n\n"
        "Com base APENAS nas informações acima (se fornecidas), gere um relatório de diagnóstico:\n"
        "1. Identificação do Problema (Hipótese)\n"
        "2. Evidência (Dados ou Manual)\n"
        "3. Ação Recomendada"
    )
    
    response_text = get_llm_response(
        req.llm_provider,
        req.llm_model,
        final_prompt,
        req.api_key,
        timeout_seconds=req.llm_timeout,
    )
    prompt_tokens = estimate_tokens(final_prompt)
    response_tokens = estimate_tokens(response_text if isinstance(response_text, str) else str(response_text))
    
    payload = {
        "response": response_text,
        "mode_used": {
            1: "Baseline (Zero-Shot)",
            2: "RAG Estático (Docs)",
            3: "RAG Dual (Docs + Telemetria)"
        }[req.scenario],
        "context_found": bool(retrieved_chunks),
        "vector_backend": vector_backend,
        "token_usage": {
            "prompt": prompt_tokens,
            "response": response_tokens,
        },
        "telemetry_signals": telemetry_keys,
    }

    if vector_debug:
        payload["vector_debug"] = vector_debug

    if req.debug:
        payload["debug"] = {
            "final_prompt": final_prompt,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_metadatas": retrieved_metadatas,
            "telemetry_used": telemetry_snapshot,
            "telemetry_signals": telemetry_keys,
            "llm_call": {
                "provider": req.llm_provider,
                "model": req.llm_model or "auto",
            },
            "prompt_sections": {
                "system": base_system,
                "instructions": req.instructions or [],
                "response_format": req.response_format or {},
                "context": context_part.strip(),
                "telemetry": telemetry_part.strip(),
                "question": req.question,
            },
        }
```

## 6. Resumo
1. PDFs entram via `/upload`, são convertidos em chunks e armazenados em múltiplos backends com metadados consistentes.
2. `query_backend` garante que a mesma infraestrutura responda às perguntas, trazendo textos e metadados alinhados.
3. `build_telemetry_section` injeta o estado dinâmico selecionado, habilitando o cenário RAG dual.
4. `run_diagnosis` monta o prompt completo, chama o LLM (Groq, Gemini ou local/Ollama) e devolve a resposta com rastreabilidade (tokens, vetores, telemetria).

Esses componentes cobrem integralmente o pipeline RAG solicitado, incluindo geração de embeddings, armazenamento no Chroma/FAISS/Weaviate/Pinecone, recuperação de contexto, montagem de prompt e fusão com telemetria.
