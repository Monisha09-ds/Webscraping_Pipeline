# ####------------------Query ----------------####
# # pipelines/query.py
# # RAG Query Pipeline (interactive CLI)

# # pipelines/query.py
# # RAG Query Pipeline (interactive CLI)

# import os
# import sys
# import argparse
# import logging
# from pathlib import Path
# from rich import print

# # Path setup
# PROJ_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJ_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJ_ROOT))

# # Project imports
# from vectorstore.store import ChromaStore
# from vectorstore.embeddings import LocalTextEmbedder
# from utils.config import CHROMA_ROOT, EMBED_MODEL_PATH
# from rag.llm import LLMWrapper  # unified naming
# from rag.answer import generate_answer

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # ----------------------
# # Helper: load titles
# # ----------------------
# def _load_index_titles(store: ChromaStore, limit: int = 20) -> list[str]:
#     titles = []
#     try:
#         for md in store.sample_metadatas(n=200):
#             t = (md.get("title") or md.get("page_title") or md.get("section") or 
#                  md.get("h_path") or md.get("source"))
#             if t and t not in titles:
#                 titles.append(t)
#             if len(titles) >= limit:
#                 break
#     except Exception:
#         pass
#     return titles[:limit]

# # ----------------------
# # Main Query Loop
# # ----------------------
# def main():
#     parser = argparse.ArgumentParser(description="Webscraper-based RAG: interactive query runner")
#     parser.add_argument("--mode", choices=["local", "api"], default=os.getenv("LLM_MODE", "api"),
#                         help="LLM backend: local HuggingFace or Google API")
#     parser.add_argument("--model-name", default=os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash"),
#                         help="API model name when --mode=api (e.g., gemini-1.5-flash)")
#     parser.add_argument("--top-k", type=int, default=8, help="Initial ANN retrieve count")
#     parser.add_argument("--debug", action="store_true", help="Print retrieved snippet previews")
#     parser.add_argument("--site-scope", default=os.getenv("SITE_SCOPE", "website-scraped corpus"),
#                         help="Short label for the corpus")
#     parser.add_argument("--session-id", default=os.getenv("SESSION_ID", "default_session"),
#                         help="Unique session identifier for state persistence")
#     args = parser.parse_args()

#     # Interactive mode selection if no arg provided
#     if not args.mode or args.mode not in ["local", "api"]:
#         args.mode = input("Select LLM mode (local/api) [default=api]: ").strip() or "api"
#     if args.mode == "api" and not args.model_name:
#         args.model_name = "gemini-1.5-flash"  # Default for API

#     print("[bold cyan]RAG Query Pipeline (ChromaDB)[/bold cyan]")
#     print("Type 'exit' to quit.\n")

#     # --- Build store and embedder ---
#     store = ChromaStore(CHROMA_ROOT)
#     embedder = LocalTextEmbedder(EMBED_MODEL_PATH, device="cpu")

#     try:
#         count = store.count()
#     except Exception:
#         count = "?"
#     print(f"Docs in DB: {count}")

#     # --- Initialize LLM ---
#     llm = LLMWrapper(mode=args.mode, model_name=args.model_name)

#     # --- Titles for rewriting ---
#     index_titles = _load_index_titles(store, limit=20)

#     # --- Session-aware chat history ---
#     global session_history
#     if 'session_history' not in globals():
#         session_history = {}
#     if args.session_id not in session_history:
#         session_history[args.session_id] = []

#     # --- Query Loop ---
#     while True:
#         try:
#             q = input("\nYour Question: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print("\nBye!")
#             break

#         if q.lower() in {"exit", "quit"}:
#             break
#         if not q:
#             continue

#         # Update chat history
#         session_history[args.session_id].append(f"User: {q}")
#         history_text = "\n".join(session_history[args.session_id][-8:])

#         # Debug mode: preview retrieval
#         if args.debug:
#             print("[dim]Fetching top chunks...[/dim]")
#             try:
#                 previews = store.search(query=q, embedder=embedder, top_k=min(args.top_k, 5))
#                 if not previews:
#                     print("[yellow]No retrieval previews found.[/yellow]")
#                 else:
#                     for i, r in enumerate(previews, 1):
#                         text = r["text"] if isinstance(r, dict) else getattr(r, "page_content", str(r))
#                         meta = (r.get("metadata") if isinstance(r, dict) else getattr(r, "metadata", {})) or {}
#                         title = meta.get("title") or meta.get("page_title") or meta.get("section") or f"Document {i}"
#                         snippet = (text or "").strip().replace("\n", " ")
#                         if len(snippet) > 160:
#                             snippet = snippet[:160].rsplit(" ", 1)[0] + " …"
#                         print(f"  • [bold]{title}[/bold]: {snippet}")
#             except Exception as e:
#                 print("[red]Preview retrieval failed:[/red]", e)

#         # Generate answer
#         try:
#             answer = generate_answer(
#                 query=q,
#                 store=store,
#                 embedder=embedder,
#                 top_k=args.top_k,
#                 llm=llm,
#             )

#             print("\n[bold green]Answer:[/bold green]\n" + (answer or "").strip())

#             # Update history with answer
#             hist_ans = answer.splitlines()[0] if answer else ""
#             session_history[args.session_id].append(f"Assistant: {hist_ans}")
#         except Exception as e:
#             print("[red]Error during answering:[/red]", e)

# # ----------------------
# # Entry Point
# # ----------------------
# if __name__ == "__main__":
#     main()



####-------------------####
# src/pipelines/query.py
import os
import sys
import argparse
import logging
from pathlib import Path
from rich import print

# Path setup
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Project imports
from vectorstore.store import ChromaStore
from vectorstore.embeddings import LocalTextEmbedder
from utils.config import CHROMA_ROOT, EMBED_MODEL_PATH
from rag.llm import LLMWrapper
from rag.answer import generate_answer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_index_titles(store: ChromaStore, limit: int = 20) -> list[str]:
    titles = []
    try:
        for md in store.sample_metadatas(n=200):
            t = (md.get("title") or md.get("page_title") or md.get("section") or
                 md.get("h_path") or md.get("source"))
            if t and t not in titles:
                titles.append(t)
            if len(titles) >= limit:
                break
    except Exception:
        pass
    return titles[:limit]


def main():
    parser = argparse.ArgumentParser(description="Webscraper-based RAG: interactive query runner")
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        help="LLM backend: 'local' for HuggingFace, 'api' for Gemini API",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash"),
        help="API model name when --mode=api (e.g., gemini-1.5-flash)",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Initial ANN retrieve count")
    parser.add_argument("--debug", action="store_true", help="Print retrieved snippet previews")
    parser.add_argument("--site-scope", default=os.getenv("SITE_SCOPE", "website-scraped corpus"))
    parser.add_argument("--session-id", default=os.getenv("SESSION_ID", "default_session"))
    args = parser.parse_args()

    # --- Interactive mode selection ---
    if not args.mode:
        while True:
            user_choice = input("Select LLM mode: [local/api] (default=api): ").strip().lower()
            if user_choice in {"local", "api"}:
                args.mode = user_choice
                break
            elif user_choice == "":
                args.mode = "api"
                break
            else:
                print("[red]Invalid choice. Please type 'local' or 'api'.[/red]")

    if args.mode == "api" and not args.model_name:
        args.model_name = "gemini-1.5-flash"

    print("[bold cyan]RAG Query Pipeline (ChromaDB)[/bold cyan]")
    print("Type 'exit' to quit.\n")

    # --- Build store and embedder ---
    store = ChromaStore(CHROMA_ROOT)
    embedder = LocalTextEmbedder(EMBED_MODEL_PATH, device="cpu")

    try:
        count = store.count()
    except Exception:
        count = "?"
    print(f"Docs in DB: {count}")

    # --- Initialize LLM ---
    llm = LLMWrapper(mode=args.mode, model_name=args.model_name)

    # --- Titles for rewriting ---
    index_titles = _load_index_titles(store, limit=20)

    # --- Session-aware chat history ---
    global session_history
    if 'session_history' not in globals():
        session_history = {}
    if args.session_id not in session_history:
        session_history[args.session_id] = []

    # --- Query Loop ---
    while True:
        try:
            q = input("\nYour Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        # Update chat history
        session_history[args.session_id].append(f"User: {q}")
        history_text = "\n".join(session_history[args.session_id][-8:])

        if args.debug:
            print("[dim]Fetching top chunks...[/dim]")
            try:
                previews = store.search(query=q, embedder=embedder, top_k=min(args.top_k, 5))
                if not previews:
                    print("[yellow]No retrieval previews found.[/yellow]")
                else:
                    for i, r in enumerate(previews, 1):
                        text = r["text"] if isinstance(r, dict) else getattr(r, "page_content", str(r))
                        meta = (r.get("metadata") if isinstance(r, dict) else getattr(r, "metadata", {})) or {}
                        title = (meta.get("title") or meta.get("page_title") or meta.get("section") or f"Document {i}")
                        snippet = (text or "").strip().replace("\n", " ")
                        if len(snippet) > 160:
                            snippet = snippet[:160].rsplit(" ", 1)[0] + " …"
                        print(f"  • [bold]{title}[/bold]: {snippet}")
            except Exception as e:
                print("[red]Preview retrieval failed:[/red]", e)

        try:
            answer = generate_answer(
                query=q,
                store=store,
                embedder=embedder,
                top_k=args.top_k,
                llm=llm,
            )
            print("\n[bold green]Answer:[/bold green]\n" + (answer or "").strip())
            hist_ans = answer.splitlines()[0] if answer else ""
            session_history[args.session_id].append(f"Assistant: {hist_ans}")
        except Exception as e:
            print("[red]Error during answering:[/red]", e)


if __name__ == "__main__":
    main()
