import os
import json
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import pypdf
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Industrial Dual-RAG API")

# --- Configurações ---
DB_DIR = os.getenv("CHROMA_DB_PATH", "/app/data/chromadb")
UPLOAD_DIR = "/app/data/uploads"
KB_Record_File = "/app/data/knowledge_base.json"

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_DIR)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="industrial_manuals", embedding_function=embedding_func)

# --- Integração LLM ---
def get_llm_response(provider: str, model: str, prompt: str, api_key: str = None):
    try:
        if provider == "groq":
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key: return "Erro: API Key da Groq não configurada."
            client = Groq(api_key=key)
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model or "llama3-8b-8192",
                temperature=0.3 # Mais determinístico para diagnósticos técnicos
            )
            return chat_completion.choices[0].message.content

        elif provider == "gemini":
            key = api_key or os.getenv("GOOGLE_API_KEY")
            if not key: return "Erro: API Key do Google não configurada."
            genai.configure(api_key=key)
            model_instance = genai.GenerativeModel(model or 'gemini-pro')
            response = model_instance.generate_content(prompt)
            return response.text
        
        elif provider == "local":
            return "[Simulação LLM Local] Diagnóstico: Falha de rolamento detectada. A vibração excede ISO 10816."
            
        else:
            return "Provedor de LLM inválido."
    except Exception as e:
        return f"Erro na chamada do LLM ({provider}): {str(e)}"

# --- Modelos ---
class ChatRequest(BaseModel):
    question: str
    scenario: int 
    telemetry: Optional[dict] = None
    llm_provider: str = "groq"
    llm_model: Optional[str] = None
    api_key: Optional[str] = None

# --- Endpoints ---

@app.get("/files")
def list_files():
    if os.path.exists(KB_Record_File):
        with open(KB_Record_File, "r") as f: return json.load(f)
    return []

@app.post("/upload")
async def upload_manual(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Apenas PDFs.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Processamento simples de PDF
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        raise HTTPException(500, f"Erro leitura PDF: {str(e)}")
    
    # Chunking (simples para o projeto)
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename} for _ in chunks]
    
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    
    # Atualiza lista
    current = list_files()
    if file.filename not in current:
        current.append(file.filename)
        with open(KB_Record_File, "w") as f: json.dump(current, f)
            
    return {"status": "indexed", "chunks": len(chunks)}

@app.post("/chat")
def run_diagnosis(req: ChatRequest):
    # Prompt Engineering Científico/Técnico
    base_system = (
        "Você é um Engenheiro Sênior de Diagnóstico Industrial. "
        "Sua tarefa é analisar falhas em máquinas rotativas. "
        "Seja técnico, direto e cite as fontes se disponíveis."
    )
    
    context_part = ""
    telemetry_part = ""
    
    # RAG Estático (Manuais)
    if req.scenario in [2, 3]:
        # Busca vetorial
        results = collection.query(query_texts=[req.question], n_results=3)
        if results['documents']:
            doc_text = "\n".join(results['documents'][0])
            context_part = f"\n=== BASE DE CONHECIMENTO (MANUAIS TÉCNICOS) ===\n{doc_text}\n"

    # RAG Dinâmico (Telemetria)
    if req.scenario == 3 and req.telemetry:
        telemetry_part = (
            f"\n=== TELEMETRIA EM TEMPO REAL (CONTEXTO DINÂMICO) ===\n"
            f"- Status Máquina: {req.telemetry.get('status', 'N/A')}\n"
            f"- Temperatura: {req.telemetry.get('temperature', 0):.1f} °C\n"
            f"- Vibração RMS: {req.telemetry.get('vibration', 0):.2f} mm/s\n"
            f"- Corrente Motor: {req.telemetry.get('current', 0):.1f} A\n"
        )
        if req.telemetry.get('temperature', 0) > 90:
            telemetry_part += "ALERTA: Temperatura acima do limite crítico.\n"
        if req.telemetry.get('vibration', 0) > 10:
            telemetry_part += "ALERTA: Vibração excessiva detectada.\n"

    # Montagem do Prompt Final
    final_prompt = (
        f"{base_system}\n"
        f"{context_part}"
        f"{telemetry_part}"
        f"\n=== SOLICITAÇÃO DO OPERADOR ===\n{req.question}\n\n"
        "Com base APENAS nas informações acima (se fornecidas), gere um relatório de diagnóstico:\n"
        "1. Identificação do Problema (Hipótese)\n"
        "2. Evidência (Dados ou Manual)\n"
        "3. Ação Recomendada"
    )
    
    response_text = get_llm_response(req.llm_provider, req.llm_model, final_prompt, req.api_key)
    
    return {
        "response": response_text,
        "mode_used": {
            1: "Baseline (Zero-Shot)",
            2: "RAG Estático (Docs)",
            3: "RAG Dual (Docs + Telemetria)"
        }[req.scenario],
        "context_found": bool(context_part)
    }