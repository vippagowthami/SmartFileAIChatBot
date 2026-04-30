from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import os
import tempfile
import re
from pathlib import Path
from typing import Any

from llm import OllamaLLM
from db import VectorDatabase
from rag import RAGPipeline
from logger import ConversationLogger
from persistence import JsonPersistence
from query_understanding import understand_query


# ------------------------------------------------------------------ #
# Component initialisation
# ------------------------------------------------------------------ #
llm = OllamaLLM(model="functiongemma", embedding_model="all-minilm")

# Auto-detect the best available Ollama model
_PREFERRED_MODELS = ["llama3", "llama3.2", "llama3.1", "mistral", "gemma", "gemma2", "phi3", "phi", "functiongemma"]
_available_models = llm.list_available_models()
if _available_models:
    _chosen = None
    for pref in _PREFERRED_MODELS:
        _match = next((m for m in _available_models if pref in m.lower()), None)
        if _match:
            _chosen = _match
            break
    if _chosen:
        llm.model = _chosen
    else:
        llm.model = _available_models[0]  # whatever is first
print(f"[startup] Using model: {llm.model}")

db = VectorDatabase(db_path="./data/chroma_db")
rag = RAGPipeline(
    llm=llm,
    db=db,
    chunk_size=1500,
    chunk_overlap=200,
    num_retrieval=5,
    relevance_threshold=0.20,
)
conversation_logger = ConversationLogger(log_dir="./logs")
indexed_files_store = JsonPersistence("./data/indexed_files.json", {"files": []})
chat_history_store = JsonPersistence("./data/chat_history.json", {"messages": []})


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _is_temp_upload_name(filename: str) -> bool:
    name = (filename or "").strip().lower()
    return bool(re.match(r"^tmp[_a-z0-9]{6,}\.(txt|pdf|doc|docx)$", name))


def _clean_indexed_files(files: list[str]) -> list[str]:
    cleaned, seen = [], set()
    for f in files:
        if not f or _is_temp_upload_name(f):
            continue
        if f not in seen:
            seen.add(f)
            cleaned.append(f)
    return cleaned


# Sync indexed-files store with what ChromaDB actually has on startup
def _sync_indexed_files_from_db() -> None:
    db_files = db.list_indexed_file_names()
    indexed_files_store.write({"files": _clean_indexed_files(db_files)})


_sync_indexed_files_from_db()

# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Smart File AI Chatbot",
    description="Local LLM + RAG + ChromaDB Chatbot",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Pydantic models
# ------------------------------------------------------------------ #
class QueryRequest(BaseModel):
    question: str
    use_rag: bool = True
    temperature: float = 0.1


class ChatHistoryPayload(BaseModel):
    messages: list[dict[str, Any]]


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_documents: int
    timings: dict
    retrieved_chunks: list = []
    retrieved_sources: list = []
    normalized_question: str | None = None
    intent: str | None = None
    verbosity: str | None = None


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Smart File AI Chatbot backend is running",
        "health": "/health",
        "query": "/query",
        "upload": "/upload",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    ollama_available = llm.is_available()
    db_stats = db.get_collection_stats()
    return {
        "status": "healthy" if ollama_available else "degraded",
        "ollama": {"available": ollama_available, "model": llm.model},
        "database": db_stats,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document into ChromaDB."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        supported = {".txt", ".pdf", ".doc", ".docx"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in supported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported: {supported}",
            )

        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            result = await run_in_threadpool(rag.ingest_file, tmp_path, file.filename)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Keep indexed-files store in sync
        current_files = indexed_files_store.read().get("files", [])
        if file.filename not in current_files:
            current_files.append(file.filename)
            indexed_files_store.write({"files": _clean_indexed_files(current_files)})

        conversation_logger.log_file_upload(
            filename=file.filename,
            num_chunks=result["chunks_created"],
            embedding_time=result["embedding_time"],
        )

        return {
            "success": True,
            "message": "File processed successfully",
            "details": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents via RAG or direct LLM."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        u = understand_query(request.question)
        if request.use_rag:
            result = await run_in_threadpool(
                rag.query,
                u.improved_query,
                request.temperature,
                intent=u.intent,
                verbosity=u.verbosity,
                original_question=request.question,
            )
        else:
            result = await run_in_threadpool(
                rag.query_without_rag,
                u.improved_query,
                request.temperature,
                intent=u.intent,
                verbosity=u.verbosity,
                original_question=request.question,
            )

        conversation_logger.log_interaction(
            question=request.question,
            answer=result["answer"],
            retrieval_time=result["timings"].get("retrieval", 0),
            generation_time=result["timings"].get("generation", 0),
            total_time=result["timings"]["total"],
            retrieved_chunks=result.get("retrieved_chunks", []),
            model=llm.model,
        )

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            retrieved_documents=result["retrieved_documents"],
            timings=result["timings"],
            retrieved_chunks=result.get("retrieved_chunks", []),
            retrieved_sources=result.get("retrieved_sources", []),
            normalized_question=result.get("normalized_question"),
            intent=u.intent,
            verbosity=u.verbosity,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    try:
        return {"status": "success", "statistics": rag.get_statistics()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-database")
async def clear_database():
    try:
        db.reset_database()
        indexed_files_store.write({"files": []})
        return {"success": True, "message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    try:
        return {"available_models": llm.list_available_models(), "current_model": llm.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indexed-files")
async def get_indexed_files():
    """Return the list of indexed files — always from ChromaDB as source of truth."""
    try:
        files = _clean_indexed_files(db.list_indexed_file_names())
        # Keep JSON store in sync
        indexed_files_store.write({"files": files})
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-history")
async def get_chat_history():
    try:
        data = chat_history_store.read()
        return {"messages": data.get("messages", []), "count": len(data.get("messages", []))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-history")
async def save_chat_history(payload: ChatHistoryPayload):
    try:
        chat_history_store.write({"messages": payload.messages})
        return {"success": True, "count": len(payload.messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Smart File AI Chatbot - FastAPI Server v2.0")
    print("=" * 60)
    print(f"Model      : {llm.model}")
    print(f"Embeddings : {llm.embedding_model}")
    print(f"Database   : {db.db_path}")
    print(f"Chunk size : {rag.chunk_size} chars  Overlap: {rag.chunk_overlap}")
    print(f"Retrieval  : top-{rag.num_retrieval}  Threshold: {rag.relevance_threshold}")
    print("Starting server on http://127.0.0.1:8001")
    print("=" * 60)

    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False, access_log=True)
