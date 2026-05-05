from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
import os
import tempfile
import re
from pathlib import Path
from typing import Any
import sys
import socket

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from backend.llm import OllamaLLM
    from backend.db import VectorDatabase
    from backend.rag import RAGPipeline
    from backend.logger import ConversationLogger
    from backend.persistence import JsonPersistence
    from backend.query_understanding import understand_query
    from backend.stt_service import STTService
    from backend.tts_service import TTSService
    from backend.wake_word import WakeWordService
else:
    from .llm import OllamaLLM
    from .db import VectorDatabase
    from .rag import RAGPipeline
    from .logger import ConversationLogger
    from .persistence import JsonPersistence
    from .query_understanding import understand_query
    from .stt_service import STTService
    from .tts_service import TTSService
    from .wake_word import WakeWordService


# ------------------------------------------------------------------ #
# Component initialisation
# ------------------------------------------------------------------ #
llm = OllamaLLM(model="functiongemma", embedding_model="all-minilm")

# Auto-detect the best available Ollama model
_PREFERRED_MODELS = ["llama3.2", "llama3", "llama3.1", "mistral", "gemma", "gemma2", "phi3", "phi", "functiongemma"]
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
    chunk_size=600,
    chunk_overlap=120,
    num_retrieval=3,
    relevance_threshold=0.45,
)
conversation_logger = ConversationLogger(log_dir="./logs")
indexed_files_store = JsonPersistence("./data/indexed_files.json", {"files": []})
chat_history_store = JsonPersistence("./data/chat_history.json", {"messages": []})
conversation_memory = {}  # Per-session memory: {session_id: [last_messages]}

# Voice layer (kept isolated from core chat pipeline)
stt_service = STTService(model_size="base")
tts_service = TTSService(engine="piper")
wake_word_service = WakeWordService(wake_word_text="Hey Smart")


def _iter_file_chunks(path: str, chunk_size: int = 64 * 1024):
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _cleanup_temp_file(path: str):
    try:
        os.unlink(path)
    except Exception:
        pass


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _get_conversation_context(session_id: str, max_messages: int = 2) -> str:
    """Retrieve last N messages for conversation continuity."""
    if session_id not in conversation_memory:
        return ""
    
    messages = conversation_memory[session_id][-max_messages:]
    if not messages:
        return ""
    
    context = "Previous conversation:\n"
    for msg in messages:
        context += f"- {msg}\n"
    return context


def _save_to_conversation_memory(session_id: str, message: str, max_memory: int = 5) -> None:
    """Save message to conversation memory."""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    conversation_memory[session_id].append(message)
    # Keep only recent messages
    if len(conversation_memory[session_id]) > max_memory:
        conversation_memory[session_id] = conversation_memory[session_id][-max_memory:]


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


@app.on_event("startup")
async def startup_event():
    """Startup tasks for the backend."""
    print("[startup] Starting Wake Word Service...")
    try:
        # Start the wake word service automatically on startup
        wake_word_service.start()
        print(f"[startup] Wake Word Service started: {wake_word_service.status()}")
    except Exception as e:
        print(f"[startup] Failed to start Wake Word Service: {e}")


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


class VoiceSettingsPayload(BaseModel):
    enable_voice_mode: bool = True
    enable_wake_word: bool = False
    wake_word_text: str = "Hey Smart"
    stt_model: str = "base"
    tts_engine: str = "piper"
    auto_play_responses: bool = True


class TTSPayload(BaseModel):
    text: str
    tts_engine: str = "piper"


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
        "voice": {
            "stt_model": stt_service.model_size,
            "tts_engine": tts_service.engine,
            "wake_word": wake_word_service.status(),
        },
    }


@app.get("/voice/settings")
async def get_voice_settings():
    return {
        "enable_voice_mode": True,
        "enable_wake_word": wake_word_service.enabled,
        "wake_word_text": wake_word_service.wake_word_text,
        "stt_model": stt_service.model_size,
        "tts_engine": tts_service.engine,
        "auto_play_responses": True,
        "wake_word_status": wake_word_service.status(),
    }


@app.post("/voice/settings")
async def set_voice_settings(payload: VoiceSettingsPayload):
    try:
        stt_service.set_model(payload.stt_model)
        tts_service.set_engine(payload.tts_engine)
        wake_status = wake_word_service.configure(
            enabled=payload.enable_wake_word and payload.enable_voice_mode,
            wake_word_text=payload.wake_word_text,
        )
        return {
            "success": True,
            "stt_model": stt_service.model_size,
            "tts_engine": tts_service.engine,
            "wake_word_status": wake_status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wake-word/events")
async def poll_wake_word_events(after_id: int = 0):
    return {"events": wake_word_service.poll(after_id=after_id)}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    stt_model: str = "base",
    language: str | None = None,
):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")

        ext = Path(file.filename).suffix.lower() or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            stt_service.set_model(stt_model)
            result = await run_in_threadpool(stt_service.transcribe, tmp_path, language)
        finally:
            _cleanup_temp_file(tmp_path)

        return {
            "text": result.text,
            "language": result.language,
            "duration_sec": result.duration_sec,
            "model": stt_service.model_size,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def synthesize_tts(payload: TTSPayload):
    try:
        if not payload.text or not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        tts_service.set_engine(payload.tts_engine)
        audio_path = await run_in_threadpool(tts_service.synthesize, payload.text)

        media_type = "audio/wav"
        return StreamingResponse(
            _iter_file_chunks(audio_path),
            media_type=media_type,
            background=BackgroundTask(_cleanup_temp_file, audio_path),
            headers={"X-TTS-Engine": tts_service.engine},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
def _find_available_port(preferred_port: int = 8000, host: str = "127.0.0.1", max_attempts: int = 20) -> int:
    port = preferred_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                port += 1
    return preferred_port


if __name__ == "__main__":
    import uvicorn

    preferred_port = int(os.getenv("PORT", "8000"))
    selected_port = _find_available_port(preferred_port)

    print("=" * 60)
    print("Smart File AI Chatbot - FastAPI Server v2.1 (Local LLM)")
    print("=" * 60)
    print(f"Model      : {llm.model}")
    print(f"Embeddings : {llm.embedding_model}")
    print(f"Database   : {db.db_path}")
    print(f"Chunk size : {rag.chunk_size} tokens  Overlap: {rag.chunk_overlap} tokens")
    print(f"Retrieval  : top-{rag.num_retrieval} (strict)  Threshold: {rag.relevance_threshold}")
    print(f"Supported formats: PDF, DOCX, TXT, CSV, JSON, XLSX")
    print(f"Starting server on http://127.0.0.1:{selected_port}")
    if selected_port != preferred_port:
        print(f"Port {preferred_port} is busy, using {selected_port} instead")
    print("=" * 60)

    uvicorn.run(app, host="127.0.0.1", port=selected_port, reload=False, access_log=True)
