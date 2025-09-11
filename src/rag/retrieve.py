# ###--------###
# # src/rag/retrieves.py

# from __future__ import annotations
# from typing import List, Dict
# import sys
# from pathlib import Path
# import logging

# PROJ_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJ_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJ_ROOT))

# from vectorstore.embeddings import LocalTextEmbedder
# from vectorstore.store import ChromaStore  

# logger = logging.getLogger(__name__)

# def retrieve_chunks(query: str, store: ChromaStore, embedder: LocalTextEmbedder, top_k: int = 5) -> List[Dict]:
#     if not query:
#         raise ValueError("Query cannot be empty")

#     try:
#         query_embedding = embedder.embed_query(query)
#         results = store.query(query_embedding, top_k=top_k)

#         if not results:
#             logger.warning("No chunks retrieved for query=%s", query)
#             return []

#         # Normalize: always return list of dicts with "text"
#         norm = []
#         for r in results:
#             if isinstance(r, str):
#                     norm.append({
#                 "text": r.get("text") or r.get("page_content") or "",
#                 "title": r.get("metadata", {}).get("title", "Unknown")
#             })

#             elif isinstance(r, dict):
#                 norm.append({"text": r.get("text") or r.get("page_content") or ""})
#             else:
#                 norm.append({"text": getattr(r, "page_content", "") or ""})
#         return norm
#     except Exception as e:
#         logger.exception("Retrieval failed: %s", e)
#         raise


# # ---------------- Local testing ----------------
# if __name__ == "__main__":
    
#     store_path = Path("/home/aiteam/Desktop/website_content_scrapper/webscraper_pipeline/chroma_store")

#     # Initialize embedder & store
#     embedder = LocalTextEmbedder() 
#     store = ChromaStore(persist_dir=store_path)

#     query = "What is deep learning?"
#     print(f"Query: {query}")

#     try:
#         results = retrieve_chunks(query, store, embedder, top_k=5)
#         print("\nTop Results:")
#         for idx, res in enumerate(results, 1):
#             print(f"{idx}. {res}")
#     except Exception as e:
#         print(f"Error during retrieval: {e}")



####---------------------------####

# rag/retrieve.py
from __future__ import annotations
from typing import List, Dict
import sys
from pathlib import Path
import logging

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from vectorstore.embeddings import LocalTextEmbedder
from vectorstore.store import ChromaStore

logger = logging.getLogger(__name__)

def _pick_title(meta: dict | None) -> str:
    meta = meta or {}
    return (
        meta.get("title")
        or meta.get("page_title")
        or meta.get("section")
        or meta.get("h_path")
        or meta.get("source")
        or "Untitled"
    )

def retrieve_chunks(
    query: str,
    store: ChromaStore,
    embedder: LocalTextEmbedder,
    top_k: int = 5
) -> List[Dict[str, str]]:
    """
    Returns a list of dicts: [{"text": "...", "title": "..."}]
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    try:
        # If your store has a convenience `search`, use it (it can call embedder inside)
        if hasattr(store, "search"):
            results = store.search(query=query, embedder=embedder, top_k=top_k)
        else:
            q_emb = embedder.embed_query(query)
            results = store.query(q_emb, top_k=top_k)

        if not results:
            logger.warning("No chunks retrieved for query=%s", query)
            return []

        normalized: List[Dict[str, str]] = []
        for r in results:
            # common shapes supported:
            # - dict: {"text": "...", "metadata": {...}}
            # - langchain Document-like: has .page_content and .metadata
            # - bare string: "...."
            if isinstance(r, dict):
                text = r.get("text") or r.get("page_content") or ""
                title = _pick_title(r.get("metadata"))
            else:
                text = getattr(r, "text", None) or getattr(r, "page_content", "") or (r if isinstance(r, str) else "")
                title = _pick_title(getattr(r, "metadata", None))

            text = (text or "").strip()
            if text:
                normalized.append({"text": text, "title": title})

        return normalized
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        raise
