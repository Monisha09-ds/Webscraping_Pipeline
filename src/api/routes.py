
####---------------####
# src/backend/api.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # NEW: CORS
from pydantic import BaseModel
import uuid
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import glob
import sys

# ---------------- Path bootstrap ----------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------- Imports from your modules ----------------
from utils.logger import get_logger
from utils.config import (
    SITECONTENT_ROOT,
    CHROMA_ROOT,
    EMBED_MODEL_PATH,
    CRAWL_STRATEGY_DEFAULT,
)
from scraper.recursion import recursive_crawl                         # CHANGED: import direct, avoid importing main.py
from vectorstore.store import build_from_sitecontent, ChromaStore     # CHANGED
from vectorstore.embeddings import LocalTextEmbedder                  # CHANGED
from rag.answer import generate_answer                                # CHANGED
from rag.llm import LLMWrapper                                        # CHANGED

app = FastAPI()
logger = get_logger("backend")
logging.basicConfig(level=logging.INFO)

# ---------------- CORS for Streamlit ----------------
# NEW: enables requests from Streamlit (usually served at 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Static folder paths (normalize) ----------------
SITECONTENT_ROOT = Path(SITECONTENT_ROOT)
CHROMA_ROOT = Path(CHROMA_ROOT)
EMBED_MODEL_PATH = Path(EMBED_MODEL_PATH)

# ---------------- In-memory session storage ----------------
# CHANGED: store real objects (store, embedder, llm) so /chat doesn't pass Paths
sessions: dict[str, dict] = {}  # session_id -> {
                                #   "folder": Path,
                                #   "collection_name": str,
                                #   "llm": LLMWrapper | None,
                                #   "store": ChromaStore | None,
                                #   "embedder": LocalTextEmbedder | None,
                                #   "chat_history": list[dict]
                                # }

# ---------------- Request Models ----------------
class ScrapeRequest(BaseModel):
    url: str
    strategy: str = CRAWL_STRATEGY_DEFAULT       # CHANGED: use config default
    max_depth: int = 2
    max_pages_per_url: int = 5
    max_total_nodes: int = 100
    max_total_pages: int = 200
    same_domain_only: bool = True

class IngestRequest(BaseModel):
    session_id: str

class ChatInitRequest(BaseModel):
    session_id: str
    model_type: str = "api"                      # "local" or "api"
    model_name: str | None = None               # NEW: allow specifying API model (e.g., gemini-1.5-flash)

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
async def read_root():
    return {"message": "Web Scraper RAG Pipeline API"}


# ---------------- Helpers (local wrappers) ----------------
def _run_crawler_sync(
    url: str,
    strategy: str,
    max_depth: int,
    max_pages_per_url: int,
    max_total_nodes: int,
    max_total_pages: int,
    same_domain_only: bool,
) -> int:
    """Sync wrapper for recursive crawler (called inside thread pool)."""
    count = recursive_crawl(
        start_url=url,
        strategy=strategy,
        max_depth=max_depth,
        max_pages_per_url=max_pages_per_url,
        max_total_pages=max_total_pages,
        max_total_nodes=max_total_nodes,
        same_domain_only=same_domain_only,
    )
    logger.info(f"Crawler done. Total pages saved: {count}")
    return count

def _run_ingest_sync(site_dir: Path, persist_dir: Path, model_path: Path, min_chars: int = 80):
    """Sync wrapper to ingest sitecontent into Chroma with embeddings."""
    if not site_dir.exists():
        raise ValueError(f"sitecontent not found: {site_dir}")
    if not any(site_dir.rglob("*.md")):
        raise ValueError(f"No .md files under: {site_dir}")
    if not model_path.exists():
        raise ValueError(f"Embedding model path not found: {model_path}")

    # smoke test embedder
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


# ---------------- /scrape ----------------
@app.post("/scrape")
async def scrape_endpoint(request: ScrapeRequest):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        try:
            session_id = str(uuid.uuid4())
            domain = request.url.split("//")[-1].split("/")[0].replace(".", "-")
            base_folder = SITECONTENT_ROOT / domain
            folder = base_folder / "home"
            folder.mkdir(parents=True, exist_ok=True)

            # Run the synchronous crawler in a background thread
            count = await loop.run_in_executor(
                pool,
                _run_crawler_sync,
                request.url,
                request.strategy,
                request.max_depth,
                request.max_pages_per_url,
                request.max_total_nodes,
                request.max_total_pages,
                request.same_domain_only,
            )
            if count == 0:
                raise ValueError("No pages were scraped. Check crawler logs for issues.")

            # Debug: list saved .md files and take the actual folder
            saved_files = list(glob.glob(str(base_folder / "**" / "*.md"), recursive=True))
            logger.info(f"Scraped files saved at: {saved_files}")
            if not saved_files:
                raise ValueError("No .md files found after scraping.")
            actual_folder = Path(saved_files[0]).parent

            collection_name = f"collection_{session_id}"
            sessions[session_id] = {
                "folder": actual_folder,
                "collection_name": collection_name,
                "llm": None,         # NEW
                "store": None,       # NEW
                "embedder": None,    # NEW
                "chat_history": [],
            }

            return {
                "session_id": session_id,
                "message": "Scraped successfully",
                "pages_saved": count,
                "folder": str(actual_folder),
            }
        except Exception as e:
            logger.error(f"Scrape failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ---------------- /ingest ----------------
@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        folder: Path = sessions[session_id]["folder"]
        logger.info(f"Starting ingest for folder: {folder}")
        _run_ingest_sync(site_dir=folder, persist_dir=CHROMA_ROOT, model_path=EMBED_MODEL_PATH, min_chars=80)
        logger.info("Vector store persisted.")
        return {"message": "Ingested successfully"}
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- /chat/init ----------------
@app.post("/chat/init")
async def chat_init(request: ChatInitRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        # CHANGED: create objects once and cache them in the session
        # LLM
        model_name = request.model_name or "gemini-1.5-flash"
        llm = LLMWrapper(mode=request.model_type, model_name=model_name)

        # Vector store + embedder
        store = ChromaStore(persist_dir=CHROMA_ROOT)
        embedder = LocalTextEmbedder(model_path=str(EMBED_MODEL_PATH), device="cpu")

        s = sessions[request.session_id]
        s["llm"] = llm
        s["store"] = store
        s["embedder"] = embedder
        s["chat_history"] = []

        return {"message": "Chat initialized"}
    except Exception as e:
        logger.error(f"Chat init failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- /chat ----------------
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        s = sessions[request.session_id]
        llm = s["llm"]
        store = s["store"]
        embedder = s["embedder"]

        if llm is None:
            raise ValueError("Chat not initialized for this session")
        if store is None or embedder is None:
            raise ValueError("Vector store or embedder not initialized for this session")

        # CHANGED: pass real objects (not Paths)
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
    uvicorn.run(app, host="0.0.0.0", port=5001)

