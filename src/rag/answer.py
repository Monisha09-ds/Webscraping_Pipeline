# # inside src/rag/answer.py

# from typing import Optional
# from pathlib import Path
# import sys

# # ---------------- Project root ----------------
# PROJ_ROOT = Path(__file__).resolve().parents[1]  # points to src/
# if str(PROJ_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJ_ROOT))

# # Local imports
# from rag.retrieves import retrieve_chunks
# from vectorstore.embeddings import LocalTextEmbedder
# from vectorstore.store import ChromaStore
# from rag.llm import LLMWrapper


# def generate_answer(query: str, store, embedder, top_k: int = 5, debug: bool = False,
#                     llm: Optional[object] = None) -> str:
#     """
#     If llm is provided, it must expose llm.generate(prompt: str, max_new_tokens: int) -> str
#     """
#     # 1) Retrieve top_k chunks
#     results = retrieve_chunks(query=query, store=store, embedder=embedder, top_k=top_k)

#     # Normalize results into text list
#     contexts = []
#     for r in results:
#         if isinstance(r, str):
#             contexts.append(r)
#         elif isinstance(r, dict):
#             contexts.append(r.get("text") or r.get("page_content") or "")
#         else:
#             contexts.append(getattr(r, "page_content", "") or "")

#     # 2) Assemble grounded prompt
#     joined_ctx = "\n\n---\n\n".join([c for c in contexts if c])
#     if not joined_ctx.strip():
#         return "The provided context does not include the answer."

#     prompt = f"""You are a helpful assistant. Answer the user query **only** using the context.
# If the context does not contain the answer, say: "The provided context does not include this information."

# Context (use citations like [#] to refer to bullet index):
# {chr(10).join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])}

# User query:
# {query}

# Instructions:
# - Be precise and concise.
# - Do not invent facts not present in Context.
# - If relevant, point to the specific [#] you used.
# - Output plain text.
# """

#     # 3) Generate
#     if llm is not None:
#         return llm.generate(prompt, max_new_tokens=300)
#     else:
#         raise RuntimeError("No LLM provided and no fallback generation path configured.")


# ---------------- Local testing ----------------
# if __name__ == "__main__":
#     # Path to your persisted Chroma store
#     # store_path = PROJ_ROOT / "chroma_store"
#     store_path = Path("/home/aiteam/Desktop/website_content_scrapper/webscraper_pipeline/chroma_store")

#     # Initialize embedder + store
#     embedder = LocalTextEmbedder()  # adapt to your setup
#     store = ChromaStore(persist_dir=store_path)

#     # Initialize LLM (local or API depending on env var LLM_MODE)
#     llm = LLMWrapper(mode="api")  # or mode="local" if you want Gemma local

#     # Example query
#     query = "Tell me the research work in the context you have?"
#     print(f"Query: {query}")

#     try:
#         answer = generate_answer(query, store, embedder, top_k=3, llm=llm)
#         print("\nGenerated Answer:")
#         print(answer)
#     except Exception as e:
#         print(f"Error: {e}")



####-------------####
# src/rag/answer.py

# rag/answer.py
from __future__ import annotations
from typing import List, Dict
import sys
from pathlib import Path
import logging

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from vectorstore.store import ChromaStore
from vectorstore.embeddings import LocalTextEmbedder
from rag.llm import LLMWrapper
from rag.prompts import build_answer_prompt  # you can extend later to add rewrite/self-check
from rag.retrieve import retrieve_chunks

logger = logging.getLogger(__name__)

def _format_context(snippets: List[Dict[str, str]]) -> str:
    """
    Turn retrieved chunks into the CONTEXT block expected by the RAG prompt.
    """
    lines = []
    for i, s in enumerate(snippets, 1):
        title = (s.get("title") or "Untitled").strip()
        text  = (s.get("text") or "").strip()
        if not text:
            continue
        # short fence with title then excerpt
        lines.append(f"[{i}] Title: {title}\nExcerpt:\n{text}\n")
    return "\n".join(lines).strip()

def generate_answer(
    query: str,
    store: ChromaStore,
    embedder: LocalTextEmbedder,
    top_k: int,
    llm: LLMWrapper
) -> str:
    """
    1) retrieve top-k chunks
    2) build the RAG prompt
    3) call LLM
    """
    if not query or not query.strip():
        return "I don't know"

    # --- Retrieve ---
    snippets = retrieve_chunks(query=query, store=store, embedder=embedder, top_k=top_k)
    if not snippets:
        logger.info("No snippets; returning unknown.")
        return "I don't know"

    # --- Build prompt ---
    context = _format_context(snippets)
    prompt = build_answer_prompt(query=query, context=context)

    # --- Generate ---
    try:
        answer = (llm.generate(prompt, max_new_tokens=512) or "").strip()
        # guardrail: enforce the contract from your SYSTEM prompt
        if not answer:
            return "I don't know"
        # If the model ignored rules and produced greetings or fluff, that’s fine for now.
        return answer
    except Exception as e:
        logger.exception("LLM generation failed: %s", e)
        return "I don't know"
