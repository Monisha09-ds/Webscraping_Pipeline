####--------------------####

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import chromadb

# ---------------- Path bootstrap ----------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------- Imports from your modules ----------------
from utils.logger import get_logger
from utils.config import (
    CHROMA_ROOT,
    EMBED_MODEL_PATH,
)
from vectorstore.store import build_from_sitecontent, ChromaStore
from vectorstore.embeddings import LocalTextEmbedder
from rag.answer import generate_answer
from rag.llm import LLMWrapper
from utils.session import load_session, save_session  # NEW: Import session helpers

app = FastAPI()
logger = get_logger("backend")
logging.basicConfig(level=logging.INFO)

# ---------------- CORS for Streamlit ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Static folder paths (normalize) ----------------
CHROMA_ROOT = Path(CHROMA_ROOT)
EMBED_MODEL_PATH = Path(EMBED_MODEL_PATH)

# ---------------- In-memory session storage ----------------
sessions: dict[str, dict] = {}  # session_id -> {
                                #   "folder": Path,
                                #   "collection_name": str,
                                #   "llm": LLMWrapper | None,
                                #   "store": ChromaStore | None,
                                #   "embedder": LocalTextEmbedder | None,
                                #   "chat_history": list[dict]
                                # }

# ---------------- Request Models ----------------
class IngestRequest(BaseModel):
    session_id: str | None = None  # Allow None for auto-generation
    folder: str

class ChatInitRequest(BaseModel):
    session_id: str
    model_type: str = "api"
    model_name: str | None = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

# ---------------- New Endpoint: /check_session ----------------
@app.get("/check_session")
async def check_session(folder: str):
    folder_path = Path(folder)
    client = chromadb.PersistentClient(path=str(CHROMA_ROOT))
    for collection in client.list_collections():
        collection_name = collection.name
        if f"collection_" in collection_name:
            session_id = collection_name.replace("collection_", "")
            if session_id in sessions and sessions[session_id].get("folder") == folder_path:
                return {"exists": True, "session_id": session_id}
    return {"exists": False}

@app.get("/")
async def read_root():
    return {"message": "Web Scraper RAG Pipeline API"}

# ---------------- Helpers (local wrappers) ----------------
def _run_ingest_sync(site_dir: Path, persist_dir: Path, model_path: Path, min_chars: int = 80):
    """Sync wrapper to ingest sitecontent into Chroma with embeddings."""
    if not site_dir.exists():
        raise ValueError(f"sitecontent not found: {site_dir}")
    if not any(site_dir.rglob("*.md")):
        raise ValueError(f"No .md files under: {site_dir}")
    if not model_path.exists():
        raise ValueError(f"Embedding model path not found: {model_path}")

    # Smoke test embedder
    embedder = LocalTextEmbedder(model_path=str(model_path), device="cpu")
    embedder.embed_documents(["hello world"])
    logger.info("Embedding model tested successfully")
    embedder.cleanup()

    store = build_from_sitecontent(
        site_dir=site_dir,
        persist_dir=persist_dir,
        embed_model_path=str(model_path),
        min_chars=min_chars,
    )
    logger.info(f"Ingest done. Docs persisted: {store.count()}")

# ---------------- /ingest ----------------
@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    folder = Path(request.folder)
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    collection_name = f"collection_{session_id}"
    logger.info(f"Starting ingest for folder: {folder} with session_id: {session_id}")

    # ... (check for existing collection)
    client = chromadb.PersistentClient(path=str(CHROMA_ROOT))
    if any(collection.name == collection_name for collection in client.list_collections()):
        logger.info(f"Embeddings already exist for session_id: {session_id}")
        sessions[session_id] = sessions.get(session_id, {
            "folder": folder,
            "collection_name": collection_name,
            "llm": None,
            "store": None,
            "embedder": None,
            "chat_history": [],
        })
        return {"message": "Embeddings already exist", "session_id": session_id}

    try:
        _run_ingest_sync(site_dir=folder, persist_dir=CHROMA_ROOT, model_path=EMBED_MODEL_PATH, min_chars=80)
        logger.info("Vector store persisted.")

        sessions[session_id] = {
            "folder": folder,
            "collection_name": collection_name,
            "llm": None,
            "store": None,
            "embedder": None,
            "chat_history": [],
        }
        logger.debug(f"Session created/updated: {sessions}")
        return {"message": "Ingested successfully", "session_id": session_id, "collection_name": collection_name}
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- /chat/init ----------------
@app.post("/chat/init")
async def chat_init(request: ChatInitRequest):
    session_id = request.session_id
    logger.debug(f"Attempting chat init for session_id: {session_id}, Sessions: {sessions}")
    if session_id not in sessions:
        logger.error(f"Session {session_id} not found in sessions: {sessions}")
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        # Create objects once and cache them in the session
        model_name = request.model_name or "gemini-1.5-flash"
        llm = LLMWrapper(mode=request.model_type, model_name=model_name)

        # Vector store + embedder (reload from collection)
        store = ChromaStore(persist_dir=CHROMA_ROOT)
        embedder = LocalTextEmbedder(model_path=str(EMBED_MODEL_PATH), device="cpu")

        s = sessions[session_id]
        s["llm"] = llm
        s["store"] = store
        s["embedder"] = embedder
        s["chat_history"] = s.get("chat_history", [])  # Reload history if previously saved

        return {"message": "Chat initialized"}
    except Exception as e:
        logger.error(f"Chat init failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- /chat ----------------
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        s = sessions[session_id]
        llm = s["llm"]
        store = s["store"]
        embedder = s["embedder"]

        if llm is None:
            raise ValueError("Chat not initialized for this session")
        if store is None or embedder is None:
            raise ValueError("Vector store or embedder not initialized for this session")

        # Pass real objects (not Paths)
        response = generate_answer(
            query=request.message,
            store=store,
            embedder=embedder,
            top_k=5,
            llm=llm,
        )

        s["chat_history"].append({"user": request.message, "assistant": response})
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)