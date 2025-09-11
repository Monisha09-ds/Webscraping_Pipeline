from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
import numpy as np
import chromadb
import os
import sys
import hashlib
import logging
from rich import print

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils.config import SITECONTENT_ROOT, CHROMA_ROOT, EMBED_MODEL_PATH
from vectorstore.embeddings import LocalTextEmbedder
from vectorstore import chunker  

logger = logging.getLogger(__name__)

class ChromaStore:
    """
    Simple ChromaDB wrapper: accepts embeddings from ingest.py and stores/retrieves them.
    """
    def __init__(self, persist_dir: Path, collection: str = "webscraper"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.col = self.client.get_or_create_collection(
            name=collection, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
    def add(self, embeddings: np.ndarray, docs: List[Dict]):
        """
        Add precomputed embeddings to ChromaDB.
        """
        self.col.upsert(
            ids=[d["id"] for d in docs],
            embeddings=[e.tolist() for e in embeddings],
            documents=[d["text"] for d in docs],
            metadatas=[d.get("meta", {}) for d in docs],
        )
        print(f"[bold green]Persisted {len(docs)} embeddings to {self.persist_dir}[/bold green]")

    def query(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Query ChromaDB with a vector. Returns top_k results.
        """
        query_emb = np.array(query_emb, dtype="float32")
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        res = self.col.query(
            query_embeddings=[query_emb[0].tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        out = []
        ids, docs, metas, dists = res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        for i in range(len(ids)):
            out.append({
                "id": ids[i],
                "text": docs[i],
                "meta": metas[i] or {},
                "score": 1.0 - float(dists[i])
            })
        return out

    def count(self) -> int:
        """Return number of documents stored in this collection."""
        return self.col.count()

def build_from_sitecontent(site_dir: Path, persist_dir: Path, embed_model_path: str, min_chars: int = 80) -> ChromaStore:
    site_dir, persist_dir = Path(site_dir), Path(persist_dir)
    store = ChromaStore(persist_dir)

    md_files = list(site_dir.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown found under {site_dir}")

    print(f"[bold cyan]Found {len(md_files)} markdown files[/bold cyan]")

    records, texts = [], []
    total_chunks = 0

    for md in md_files:
        txt = md.read_text(encoding="utf-8", errors="ignore").strip()
        if len(txt) < min_chars:
            print(f"[yellow]Skipping {md.name} (too short)[/yellow]")
            continue

        url, title = "", ""
        meta_path = md.parent / "_meta.json"
        if meta_path.exists():
            try:
                jd = json.loads(meta_path.read_text(encoding="utf-8"))
                url = jd.get("final_url") or jd.get("url") or ""
                title = jd.get("title") or ""
            except Exception:
                pass

        doc_id = hashlib.md5(str(md.resolve()).encode()).hexdigest()
        chunks = chunker.chunk_markdown(txt, doc_id=doc_id, source=url or md.as_uri())

        for ch in chunks:
            records.append({
                "id": ch.metadata["chunk_id"],
                "text": ch.text,
                "meta": ch.metadata
            })
            texts.append(ch.text)

        print(f"[green]Processed {md.name} → {len(chunks)} chunks[/green]")
        total_chunks += len(chunks)

    if not texts:
        raise RuntimeError("Zero chunks produced; aborting.")

    print(f"[bold cyan]Total chunks produced: {total_chunks}[/bold cyan]")

    encoder = LocalTextEmbedder(str(embed_model_path), device="cpu")
    vecs = encoder.embed_documents(texts)
    embedding_dim = len(vecs[0]) if vecs else 0
    encoder.cleanup()

    (persist_dir / "stats.json").write_text(json.dumps({
        "model": str(embed_model_path),
        "dim": embedding_dim,
        "count": len(records),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }, indent=2))

    store.add(np.array(vecs), records)
    return store

