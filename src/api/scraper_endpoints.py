# backend.py - FastAPI backend for the web scraper (SCRAPER-ONLY)

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import glob
import sys

PROJ_ROOT = Path(__file__).resolve().parents[2]      # -> .../webscraper_pipeline
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.logger import get_logger
from utils.config import SITECONTENT_ROOT, CHROMA_ROOT  # single source of truth
from scraper.recursion import recursive_crawl

app = FastAPI()
logger = get_logger("scraper-backend")
logging.basicConfig(level=logging.INFO)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- In-memory session storage ----------------
# session_id -> {"folder": Path, "collection_name": str}
sessions: dict[str, dict] = {}

class ScrapeRequest(BaseModel):
    url: str
    strategy: str = "bfs"           # or "dfs"
    max_depth: int = 2
    max_pages_per_url: int = 5
    max_total_nodes: int = 100
    max_total_pages: int = 200
    same_domain_only: bool = True

class IngestRequest(BaseModel):
    session_id: str  # (kept if you later add /ingest in this service)

def _run_crawler_sync(
    url: str,
    strategy: str,
    max_depth: int,
    max_pages_per_url: int,
    max_total_nodes: int,
    max_total_pages: int,
    same_domain_only: bool,
) -> int:
    """
    Synchronous wrapper around the recursive crawler.
    """
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

@app.get("/")
async def read_root():
    return {"message": "Scraper API is alive"}

@app.post("/scrape")
async def scrape_endpoint(request: ScrapeRequest):
    """
    Kick off a crawl and return a new session_id plus the folder where .md files were saved.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        try:
            session_id = str(uuid.uuid4())
            # Normalize domain to folder name (replace dots with dashes)
            domain = request.url.split("//")[-1].split("/")[0].replace(".", "-")
            base_folder = Path(SITECONTENT_ROOT) / domain
            # Your crawler saves under {domain}/home; keep that convention
            folder = base_folder / "home"
            folder.mkdir(parents=True, exist_ok=True)

            # Run crawler off-thread
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

            # Find saved files; derive the actual folder saved by the saver
            saved_files = list(glob.glob(str(base_folder / "**" / "*.md"), recursive=True))
            logger.info(f"Scraped files saved at: {saved_files}")
            if not saved_files:
                raise ValueError("No .md files found after scraping.")
            actual_folder = Path(saved_files[0]).parent

            collection_name = f"collection_{session_id}"
            sessions[session_id] = {
                "folder": actual_folder,
                "collection_name": collection_name,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)
