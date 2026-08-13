# src/rag/answer.py
#
# Retrieve -> build grounded prompt -> generate.
#
# "I don't know" is reserved for the case where retrieval genuinely returned
# nothing relevant. Operational failures (bad API key, quota, network) raise
# LLMError so the caller can surface a real error instead of a fake answer.

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils.config import RETRIEVAL_TOP_K
from rag.llm import LLMError, LLMWrapper
from rag.prompts import build_answer_prompt
from rag.retrieve import retrieve_chunks
from vectorstore.embeddings import LocalTextEmbedder
from vectorstore.store import ChromaStore

logger = logging.getLogger(__name__)

NO_ANSWER = "I don't know"


def _format_context(snippets: List[Dict[str, str]]) -> str:
    """Turn retrieved chunks into the CONTEXT block the RAG prompt expects."""
    lines = []
    for i, s in enumerate(snippets, 1):
        title = (s.get("title") or "Untitled").strip()
        text = (s.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{i}] Title: {title}\nExcerpt:\n{text}\n")
    return "\n".join(lines).strip()


def generate_answer(
    query: str,
    store: ChromaStore,
    embedder: LocalTextEmbedder,
    top_k: int = RETRIEVAL_TOP_K,
    llm: LLMWrapper | None = None,
) -> str:
    if not query or not query.strip():
        return NO_ANSWER
    if llm is None:
        raise ValueError("An LLMWrapper instance is required")

    snippets = retrieve_chunks(query=query, store=store, embedder=embedder, top_k=top_k)
    if not snippets:
        logger.info("No snippets retrieved; answering %r", NO_ANSWER)
        return NO_ANSWER

    prompt = build_answer_prompt(query=query, context=_format_context(snippets))

    answer = (llm.generate(prompt, max_new_tokens=512) or "").strip()
    if not answer:
        # The model returned nothing at all - that is a generation problem, not
        # a missing-context problem.
        raise LLMError("The model returned an empty response.")
    return answer
