from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import chromadb
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils.config import CHROMA_ROOT, COLLECTION_PREFIX, EMBED_MODEL_NAME, INGEST_MIN_CHARS
from vectorstore import chunker
from vectorstore.embeddings import LocalTextEmbedder

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    ChromaDB wrapper. One collection per scraped site so two sites never share a
    namespace and a query can never retrieve another site's chunks.
    """

    def __init__(self, persist_dir: Path = CHROMA_ROOT, collection: str = COLLECTION_PREFIX):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.col = self.client.get_or_create_collection(
            name=collection,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------------- write ----------------
    def add(self, embeddings: Sequence, docs: List[Dict]) -> None:
        """Store precomputed embeddings alongside their documents."""
        if not docs:
            return
        self.col.upsert(
            ids=[d["id"] for d in docs],
            embeddings=[
                e.tolist() if isinstance(e, np.ndarray) else list(e) for e in embeddings
            ],
            documents=[d["text"] for d in docs],
            metadatas=[d.get("meta", {}) for d in docs],
        )
        logger.info(
            "Persisted %d embeddings to %s (collection=%s)",
            len(docs), self.persist_dir, self.collection_name,
        )

    # ---------------- read ----------------
    def query(self, query_emb, top_k: int = 5) -> List[Dict]:
        """Vector search. Returns dicts carrying both `meta` and `metadata`."""
        vec = np.asarray(query_emb, dtype="float32")
        if vec.ndim == 1:
            vec = vec[None, :]

        res = self.col.query(
            query_embeddings=[vec[0].tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not res.get("ids") or not res["ids"][0]:
            return []

        ids, docs = res["ids"][0], res["documents"][0]
        metas, dists = res["metadatas"][0], res["distances"][0]

        out = []
        for i in range(len(ids)):
            meta = metas[i] or {}
            out.append({
                "id": ids[i],
                "text": docs[i],
                # Both keys are populated: callers historically read either one.
                "meta": meta,
                "metadata": meta,
                "score": 1.0 - float(dists[i]),
            })
        return out

    def search(self, query: str, embedder: LocalTextEmbedder, top_k: int = 5) -> List[Dict]:
        """Embed `query` with the query-side prefix, then search."""
        return self.query(embedder.embed_query(query), top_k=top_k)

    def count(self) -> int:
        return self.col.count()

    def delete(self) -> None:
        """Drop this collection entirely (used when re-ingesting a site)."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted collection %s", self.collection_name)
        except Exception as e:  # collection may not exist yet
            logger.debug("delete_collection(%s) skipped: %s", self.collection_name, e)


def collection_exists(persist_dir: Path, collection: str) -> bool:
    """True if `collection` is already present and non-empty."""
    client = chromadb.PersistentClient(path=str(Path(persist_dir)))
    names = {c.name for c in client.list_collections()}
    if collection not in names:
        return False
    return client.get_collection(collection).count() > 0


def build_from_sitecontent(
    site_dir: Path,
    persist_dir: Path = CHROMA_ROOT,
    embed_model_path: str = EMBED_MODEL_NAME,
    min_chars: int = INGEST_MIN_CHARS,
    collection: str = COLLECTION_PREFIX,
) -> ChromaStore:
    """Chunk every markdown file under `site_dir` and persist it to `collection`."""
    site_dir, persist_dir = Path(site_dir), Path(persist_dir)
    store = ChromaStore(persist_dir, collection=collection)

    md_files = list(site_dir.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown found under {site_dir}")

    logger.info("Found %d markdown files under %s", len(md_files), site_dir)

    records: List[Dict] = []
    texts: List[str] = []

    for md in md_files:
        txt = md.read_text(encoding="utf-8", errors="ignore").strip()
        if len(txt) < min_chars:
            logger.debug("Skipping %s (only %d chars)", md.name, len(txt))
            continue

        url, title = "", ""
        meta_path = md.parent / "_meta.json"
        if meta_path.exists():
            try:
                jd = json.loads(meta_path.read_text(encoding="utf-8"))
                url = jd.get("final_url") or jd.get("url") or ""
                title = jd.get("title") or ""
            except (OSError, json.JSONDecodeError):
                pass

        doc_id = hashlib.md5(str(md.resolve()).encode()).hexdigest()
        chunks = chunker.chunk_markdown(txt, doc_id=doc_id, source=url or md.as_uri())

        for ch in chunks:
            meta = dict(ch.metadata)
            # Carry the page title through so retrieval can cite it.
            if title:
                meta.setdefault("title", title)
            meta.setdefault("title", ch.section_title or "Untitled")
            records.append({"id": meta["chunk_id"], "text": ch.text, "meta": meta})
            texts.append(ch.text)

        logger.info("Processed %s -> %d chunks", md.name, len(chunks))

    if not texts:
        raise RuntimeError("Zero chunks produced; aborting.")

    logger.info("Total chunks produced: %d", len(texts))

    encoder = LocalTextEmbedder(str(embed_model_path))
    vecs = encoder.embed_documents(texts)
    embedding_dim = len(vecs[0]) if vecs else 0
    encoder.cleanup()

    store.add(vecs, records)

    stats_path = persist_dir / f"stats_{collection}.json"
    stats_path.write_text(
        json.dumps(
            {
                "collection": collection,
                "model": str(embed_model_path),
                "dim": embedding_dim,
                "count": len(records),
                "source_dir": str(site_dir),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return store
