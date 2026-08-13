# src/api/chat_api.py - FastAPI service for ingest + RAG chat.

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- Path bootstrap ----------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import (
    CHAT_API_PORT,
    CHROMA_ROOT,
    CORS_ORIGINS,
    EMBED_MODEL_NAME,
    GROQ_MODEL,
    GROQ_MODEL_CHOICES,
    INGEST_MIN_CHARS,
    LLM_PROVIDER,
    RETRIEVAL_TOP_K,
    collection_for_session,
)
from utils.logger import get_logger
from utils.session import (
    list_sessions,
    load_session,
    save_session,
    session_id_for_folder,
)
from rag.answer import generate_answer
from rag.llm import LLMError, LLMWrapper
from vectorstore.embeddings import LocalTextEmbedder
from vectorstore.store import ChromaStore, build_from_sitecontent, collection_exists

logger = get_logger("chat-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Webscraper RAG - Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Runtime state ----------------
# Live objects only. Anything that must survive a restart goes to disk via
# utils.session, and is rehydrated on demand by _get_session().
sessions: dict[str, dict[str, Any]] = {}

# One embedder for the whole process. Every session uses the same model, so
# loading a copy per session would waste hundreds of MB for no benefit.
_embedder: LocalTextEmbedder | None = None


def get_embedder() -> LocalTextEmbedder:
    global _embedder
    if _embedder is None:
        logger.info("Loading shared embedder: %s", EMBED_MODEL_NAME)
        _embedder = LocalTextEmbedder()
    return _embedder


def _get_session(session_id: str) -> dict[str, Any]:
    """Fetch a session from memory, falling back to disk. 404 if unknown."""
    if session_id in sessions:
        return sessions[session_id]

    persisted = load_session(session_id)
    if persisted is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found. Ingest a scraped folder first.",
        )

    logger.info("Rehydrating session %s from disk", session_id)
    sessions[session_id] = {
        "folder": Path(persisted["folder"]) if persisted.get("folder") else None,
        "collection_name": persisted.get("collection_name")
        or collection_for_session(session_id),
        "chat_history": persisted.get("chat_history", []),
        "llm": None,
        "store": None,
    }
    return sessions[session_id]


# ---------------- Request models ----------------
class IngestRequest(BaseModel):
    folder: str
    session_id: str | None = None
    force: bool = False


class ChatInitRequest(BaseModel):
    session_id: str
    model_type: str | None = None          # "groq" | "local" (legacy: "api")
    model_name: str | None = None
    api_key: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


# ---------------- Meta endpoints ----------------
@app.get("/")
async def read_root():
    return {"message": "Web Scraper RAG Pipeline API"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "llm_model": GROQ_MODEL,
        "llm_model_choices": GROQ_MODEL_CHOICES,
        "embed_model": EMBED_MODEL_NAME,
        "active_sessions": len(sessions),
    }


@app.get("/sessions")
async def get_sessions():
    return {"sessions": list_sessions()}


@app.get("/check_session")
async def check_session(folder: str):
    """
    Does this scraped folder already have embeddings?

    The id is derived from the folder path, so the answer is identical across
    processes and survives a restart.
    """
    session_id = session_id_for_folder(folder)
    collection = collection_for_session(session_id)
    exists = collection_exists(CHROMA_ROOT, collection)
    return {
        "exists": exists,
        "session_id": session_id,
        "collection_name": collection,
    }


# ---------------- /ingest ----------------
@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    folder = Path(request.folder)
    session_id = request.session_id or session_id_for_folder(folder)
    collection = collection_for_session(session_id)

    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
    if not any(folder.rglob("*.md")):
        raise HTTPException(status_code=400, detail=f"No .md files under: {folder}")

    already = collection_exists(CHROMA_ROOT, collection)
    if already and not request.force:
        logger.info("Embeddings already exist for %s (collection=%s)", session_id, collection)
        state = sessions.setdefault(
            session_id,
            {
                "folder": folder,
                "collection_name": collection,
                "chat_history": [],
                "llm": None,
                "store": None,
            },
        )
        save_session(session_id, state)
        return {
            "message": "Embeddings already exist",
            "session_id": session_id,
            "collection_name": collection,
        }

    logger.info("Ingesting %s -> collection=%s", folder, collection)
    try:
        if already and request.force:
            ChromaStore(CHROMA_ROOT, collection=collection).delete()

        store = build_from_sitecontent(
            site_dir=folder,
            persist_dir=CHROMA_ROOT,
            embed_model_path=EMBED_MODEL_NAME,
            min_chars=INGEST_MIN_CHARS,
            collection=collection,
        )
        chunk_count = store.count()
        logger.info("Ingest complete. Chunks persisted: %d", chunk_count)
    except Exception as e:
        logger.error("Ingest failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    state = {
        "folder": folder,
        "collection_name": collection,
        "chat_history": [],
        "embed_model": EMBED_MODEL_NAME,
        "llm": None,
        "store": None,
    }
    sessions[session_id] = state
    save_session(session_id, state)

    return {
        "message": "Ingested successfully",
        "session_id": session_id,
        "collection_name": collection,
        "chunks": chunk_count,
    }


# ---------------- /chat/init ----------------
@app.post("/chat/init")
async def chat_init(request: ChatInitRequest):
    state = _get_session(request.session_id)
    collection = state.get("collection_name") or collection_for_session(request.session_id)

    if not collection_exists(CHROMA_ROOT, collection):
        raise HTTPException(
            status_code=409,
            detail=f"No embeddings found for this session (collection={collection}). Run /ingest first.",
        )

    try:
        llm = LLMWrapper(
            provider=request.model_type or LLM_PROVIDER,
            model_name=request.model_name,
            api_key=request.api_key,
        )
    except LLMError as e:
        # Missing/invalid key is the user's problem to fix, not a server fault.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Chat init failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    state["llm"] = llm
    state["store"] = ChromaStore(CHROMA_ROOT, collection=collection)
    state["collection_name"] = collection
    state.setdefault("chat_history", [])

    # Warm the shared embedder now so the first question is not slow.
    get_embedder()

    return {
        "message": "Chat initialized",
        "session_id": request.session_id,
        "provider": llm.provider,
        "model": llm.model_name,
        "collection_name": collection,
        "chunks": state["store"].count(),
    }


# ---------------- /chat ----------------
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    state = _get_session(request.session_id)

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if state.get("llm") is None or state.get("store") is None:
        raise HTTPException(
            status_code=409,
            detail="Chat not initialized for this session. Call /chat/init first.",
        )

    try:
        response = generate_answer(
            query=request.message,
            store=state["store"],
            embedder=get_embedder(),
            top_k=RETRIEVAL_TOP_K,
            llm=state["llm"],
        )
    except LLMError as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM error: {e}") from e
    except Exception as e:
        logger.exception("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    state["chat_history"].append({"user": request.message, "assistant": response})
    save_session(request.session_id, state)
    return {"response": response, "session_id": request.session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=CHAT_API_PORT)
