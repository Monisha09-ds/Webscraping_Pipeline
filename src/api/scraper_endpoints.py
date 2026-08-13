# src/api/scraper_endpoints.py - FastAPI service for the recursive crawler.

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

PROJ_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import (
    CORS_ORIGINS,
    CRAWL_STRATEGY_DEFAULT,
    SCRAPER_API_PORT,
    SITECONTENT_ROOT,
    collection_for_session,
)
from utils.logger import get_logger
from utils.session import save_session, session_id_for_folder
from scraper.recursion import recursive_crawl

logger = get_logger("scraper-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Webscraper RAG - Scraper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScrapeRequest(BaseModel):
    url: str
    strategy: str = CRAWL_STRATEGY_DEFAULT
    max_depth: int = 2
    max_pages_per_url: int = 5
    max_total_nodes: int = 100
    max_total_pages: int = 200
    same_domain_only: bool = True

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        parsed = urlparse(v.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute http(s) URL, e.g. https://example.com")
        return v.strip()

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        if v not in {"bfs", "dfs"}:
            raise ValueError("strategy must be 'bfs' or 'dfs'")
        return v


def _run_crawler_sync(
    url: str,
    strategy: str,
    max_depth: int,
    max_pages_per_url: int,
    max_total_nodes: int,
    max_total_pages: int,
    same_domain_only: bool,
) -> int:
    count = recursive_crawl(
        start_url=url,
        strategy=strategy,
        max_depth=max_depth,
        max_pages_per_url=max_pages_per_url,
        max_total_pages=max_total_pages,
        max_total_nodes=max_total_nodes,
        same_domain_only=same_domain_only,
    )
    logger.info("Crawler done. Total pages saved: %d", count)
    return count


@app.get("/")
async def read_root():
    return {"message": "Scraper API is alive"}


@app.get("/health")
async def health():
    return {"status": "ok", "sitecontent_root": str(SITECONTENT_ROOT)}


@app.post("/scrape")
async def scrape_endpoint(request: ScrapeRequest):
    """
    Crawl a site into sitecontent/ and return the folder plus the session id the
    chat service will use for it.

    The id is derived from the output folder, so the chat API arrives at the same
    id independently - no state is shared between the two services.
    """
    loop = asyncio.get_running_loop()
    domain = urlparse(request.url).netloc.replace(".", "-").replace(":", "-")
    base_folder = Path(SITECONTENT_ROOT) / domain
    base_folder.mkdir(parents=True, exist_ok=True)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
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
            raise ValueError("No pages were scraped. Check the crawler logs for details.")

        saved_files = sorted(base_folder.rglob("*.md"))
        if not saved_files:
            raise ValueError("No .md files found after scraping.")

        # The crawler nests output; ingest the common root so every page is picked up.
        actual_folder = base_folder
        logger.info("Scraped %d files under %s", len(saved_files), actual_folder)

        session_id = session_id_for_folder(actual_folder)
        collection = collection_for_session(session_id)
        save_session(
            session_id,
            {"folder": actual_folder, "collection_name": collection, "chat_history": []},
        )

        return {
            "session_id": session_id,
            "message": "Scraped successfully",
            "pages_saved": count,
            "files_found": len(saved_files),
            "folder": str(actual_folder),
            "collection_name": collection,
        }
    except ValueError as e:
        logger.error("Scrape failed: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Scrape failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=SCRAPER_API_PORT)
