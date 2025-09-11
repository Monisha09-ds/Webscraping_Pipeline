# search.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict

import os 
import sys

PROJ_ROOT = Path(__file__).resolve().parents[1]            
if str(PROJ_ROOT) not in sys.path: sys.path.insert(0, str(PROJ_ROOT))

from pathlib import Path
from typing import List, Dict
from vectorstore.embeddings import LocalTextEmbedder
from vectorstore.store import ChromaStore

def quick_retrieve(persist_dir: Path, question: str, embed_model_path: str, top_k: int = 5) -> List[Dict]:
    """
    Convenience wrapper:
    - Embeds the query string
    - Opens ChromaStore
    - Returns top_k results
    """
    enc = LocalTextEmbedder(embed_model_path, device="cpu")
    qvec = enc.embed_query(question)
    enc.cleanup()
    store = ChromaStore(persist_dir)
    return store.query(qvec, top_k=top_k)
