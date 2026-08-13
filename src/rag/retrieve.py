# src/rag/retrieve.py

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils.config import RETRIEVAL_TOP_K
from vectorstore.embeddings import LocalTextEmbedder
from vectorstore.store import ChromaStore

logger = logging.getLogger(__name__)


def _pick_title(meta: dict | None) -> str:
    meta = meta or {}
    for key in ("title", "page_title", "section", "h_path", "source"):
        value = meta.get(key)
        if value:
            return str(value)
    return "Untitled"


def _extract_meta(result) -> dict:
    """Chroma results carry `meta`; Document-like objects carry `metadata`."""
    if isinstance(result, dict):
        return result.get("meta") or result.get("metadata") or {}
    return getattr(result, "metadata", None) or {}


def retrieve_chunks(
    query: str,
    store: ChromaStore,
    embedder: LocalTextEmbedder,
    top_k: int = RETRIEVAL_TOP_K,
) -> List[Dict[str, str]]:
    """Returns [{"text": ..., "title": ..., "score": ...}] ordered by relevance."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    try:
        if hasattr(store, "search"):
            results = store.search(query=query, embedder=embedder, top_k=top_k)
        else:
            results = store.query(embedder.embed_query(query), top_k=top_k)
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        raise

    if not results:
        logger.info("No chunks retrieved for query=%r", query)
        return []

    normalized: List[Dict[str, str]] = []
    for r in results:
        if isinstance(r, dict):
            text = r.get("text") or r.get("page_content") or ""
            score = r.get("score")
        elif isinstance(r, str):
            text, score = r, None
        else:
            text = getattr(r, "text", None) or getattr(r, "page_content", "") or ""
            score = getattr(r, "score", None)

        text = (text or "").strip()
        if not text:
            continue

        entry: Dict[str, str] = {"text": text, "title": _pick_title(_extract_meta(r))}
        if score is not None:
            entry["score"] = score
        normalized.append(entry)

    return normalized
