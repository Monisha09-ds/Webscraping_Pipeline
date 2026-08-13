# src/vectorstore/embeddings.py
#
# Text embedder backed by sentence-transformers.
#
# Two things this handles that a raw AutoModel + mean-pool does not:
#   1. Attention-masked pooling, so padding tokens never leak into a vector.
#   2. Asymmetric task prefixes (nomic-embed needs "search_document: " on docs
#      and "search_query: " on queries), applied consistently on both sides.

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils.config import (
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    EMBED_DOC_PREFIX,
    EMBED_MAX_SEQ_LENGTH,
    EMBED_MODEL_NAME,
    EMBED_QUERY_PREFIX,
    EMBED_TRUST_REMOTE_CODE,
)

logger = logging.getLogger(__name__)


def _resolve_device(device: Optional[str]) -> str:
    """Turn 'auto'/None into a concrete torch device string."""
    requested = (device or EMBED_DEVICE or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class LocalTextEmbedder:
    """
    Embedder for scraped markdown content.

    Accepts either a HuggingFace model id (downloaded and cached on first use)
    or a path to a local model directory.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        doc_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = str(model_path or EMBED_MODEL_NAME)
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.doc_prefix = EMBED_DOC_PREFIX if doc_prefix is None else doc_prefix
        self.query_prefix = EMBED_QUERY_PREFIX if query_prefix is None else query_prefix

        logger.info("[Embeddings] Loading %s on %s", self.model_name, self.device)
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=EMBED_TRUST_REMOTE_CODE,
        )

        # Cap the window: scraped pages are long and the default can blow up RAM.
        if EMBED_MAX_SEQ_LENGTH:
            self.model.max_seq_length = min(
                EMBED_MAX_SEQ_LENGTH, self.model.max_seq_length or EMBED_MAX_SEQ_LENGTH
            )

        self.dimension = int(self.model.get_sentence_embedding_dimension())
        logger.info(
            "[Embeddings] Ready: dim=%d, max_seq_length=%s, doc_prefix=%r, query_prefix=%r",
            self.dimension,
            self.model.max_seq_length,
            self.doc_prefix,
            self.query_prefix,
        )

    def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> List[List[float]]:
        """Embed a list of passages. Returns plain lists so Chroma can store them."""
        if not texts:
            return []

        prefixed = [f"{self.doc_prefix}{t}" for t in texts]
        vectors = self.model.encode(
            prefixed,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        logger.info("[Embeddings] Generated %d vectors of size %d", len(vectors), self.dimension)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """Embed a single user query, using the query-side prefix."""
        vector = self.model.encode(
            f"{self.query_prefix}{query}",
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def cleanup(self) -> None:
        """Release the model so a long-lived process does not hold the RAM."""
        logger.info("[Embeddings] Cleaning up %s", self.model_name)
        self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
