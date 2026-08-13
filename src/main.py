# main.py - Orchestrator for crawler, ingest, inspect, and chat (query) pipelines

from __future__ import annotations
import sys
from pathlib import Path
from urllib.parse import urlparse
import argparse

# Path bootstrap
PROJ_ROOT = Path(__file__).resolve().parents[1]  # Assuming main.py is in pipelines/
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Imports from the scripts
from utils.logger import get_logger
from scraper.recursion import recursive_crawl
from utils.config import (
    CRAWL_STRATEGY_DEFAULT,
    SITECONTENT_ROOT,
    CHROMA_ROOT,
    COLLECTION_PREFIX,
    EMBED_MODEL_NAME,
    GROQ_MODEL,
    LLM_PROVIDER,
    collection_for_session,
)
from utils.session import session_id_for_folder
from vectorstore.store import build_from_sitecontent, ChromaStore
from vectorstore.embeddings import LocalTextEmbedder

# RAG components
from rag.answer import generate_answer
from rag.llm import LLMWrapper

log = get_logger("orchestrator")


# ---------------- Utils ----------------
def ask_int(prompt: str, default: int) -> int:
    try:
        v = input(f"{prompt} [default={default}]: ").strip()
        return int(v) if v else default
    except ValueError:
        return default


def ask_bool(prompt: str, default_yes=True) -> bool:
    v = input(f"{prompt} [{'Y/n' if default_yes else 'y/N'}]: ").strip().lower()
    if not v:
        return default_yes
    return v in ("y", "yes")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    pretty = "/".join(choices)
    v = input(f"{prompt} [{pretty}] [default={default}]: ").strip().lower()
    return v if v in choices else default


# ---------------- Core Functions ----------------
def run_crawler(
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
    log.info(f"Crawler done. Total pages saved: {count}")
    return count


def run_ingest(
    site_dir: Path,
    persist_dir: Path,
    embed_model: str,
    min_chars: int,
    collection: str,
):
    if not site_dir.exists():
        raise ValueError(f"sitecontent not found: {site_dir}")
    if not any(site_dir.rglob("*.md")):
        raise ValueError(f"No .md files under: {site_dir}")

    store = build_from_sitecontent(
        site_dir=site_dir,
        persist_dir=persist_dir,
        embed_model_path=embed_model,
        min_chars=min_chars,
        collection=collection,
    )
    log.info(f"Ingest done. Chunks persisted: {store.count()} (collection={collection})")


def run_inspect(persist_dir: Path, collection: str):
    store = ChromaStore(persist_dir=persist_dir, collection=collection)
    log.info(f"Chunks in collection {collection}: {store.count()}")
    log.info(f"Persistence dir: {persist_dir}")


# --------- NEW: interactive chat (query) ----------
def run_chat(
    persist_dir: Path,
    collection: str,
    embed_model: str,
    top_k: int = 3,
    provider_arg: str | None = None,
    model_name_arg: str | None = None,
):
    """
    Opens a CLI chat after DB is ready. Lets the user pick the LLM backend.
    """
    # vector store + embedder
    store = ChromaStore(persist_dir=persist_dir, collection=collection)
    embedder = LocalTextEmbedder(model_path=embed_model)

    provider = provider_arg or ask_choice(
        "Select LLM provider", ["groq", "local"], default=LLM_PROVIDER
    )

    if provider == "groq":
        default_model = model_name_arg or GROQ_MODEL
        user_model = input(f"Groq model [default={default_model}]: ").strip() or default_model
        llm = LLMWrapper(provider="groq", model_name=user_model)
        log.info(f"Chat will use Groq model: {user_model}")
    else:
        log.info("Chat will use a local HuggingFace model.")
        llm = LLMWrapper(provider="local", model_name=model_name_arg)

    log.info("Chat mode started. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYour Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        try:
            answer = generate_answer(
                query=q,
                store=store,
                embedder=embedder,
                top_k=top_k,
                llm=llm,
            )
            print("\n[Answer]\n" + (answer or "").strip())
        except Exception as e:
            print("[Error during answering:]", e)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate crawler, ingest, inspect, and chat pipelines"
    )
    parser.add_argument("--url", type=str, help="Start URL for crawling (required for full run)")
    parser.add_argument(
        "--strategy",
        type=str,
        default=CRAWL_STRATEGY_DEFAULT,
        choices=["bfs", "dfs"],
        help="Crawl strategy",
    )
    parser.add_argument("--max-depth", type=int, default=2, help="Max recursion depth")
    parser.add_argument("--max-pages-per-url", type=int, default=5, help="Max pages per URL")
    parser.add_argument("--max-total-nodes", type=int, default=100, help="Global node cap")
    parser.add_argument("--max-total-pages", type=int, default=200, help="Global page cap")
    parser.add_argument(
        "--same-domain-only",
        action="store_true",
        default=True,
        help="Restrict to same domain",
    )
    parser.add_argument(
        "--sitecontent", type=Path, default=SITECONTENT_ROOT, help="Sitecontent folder"
    )
    parser.add_argument("--persist", type=Path, default=CHROMA_ROOT, help="Chroma persist dir")
    parser.add_argument(
        "--embed-model",
        type=str,
        default=EMBED_MODEL_NAME,
        help="Embedding model id or local path",
    )
    parser.add_argument("--min-chars", type=int, default=80, help="Min chars per file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode for inputs")

    # Chat / query flags
    parser.add_argument("--chat", action="store_true", help="Enter chat mode after pipelines")
    parser.add_argument("--top-k", type=int, default=3, help="Top K chunks for retrieval")
    parser.add_argument(
        "--llm-provider",
        choices=["groq", "local"],
        help="Force the LLM provider for chat (otherwise asked interactively)",
    )
    parser.add_argument(
        "--llm-model-name",
        help=f"Model name for the chosen provider (e.g. {GROQ_MODEL}).",
    )

    args = parser.parse_args()

    # Interactive or non-interactive parameter collection
    if args.interactive or not args.url:
        url = input("Enter the START URL to crawl: ").strip()
        if not url:
            log.error("No URL provided.")
            return
        strategy = ask_choice("Crawl strategy", ["bfs", "dfs"], args.strategy)
        max_depth = ask_int("Max recursion depth (0 = only start page)", args.max_depth)
        max_pages_per_url = ask_int("Max pages per URL (pagination cap)", args.max_pages_per_url)
        max_total_nodes = ask_int("Global node cap (unique URLs)", args.max_total_nodes)
        max_total_pages = ask_int("Global page cap (safety limit)", args.max_total_pages)
        same_domain_only = ask_bool(
            "Restrict to same registrable/host scope? (recommended: Y)", args.same_domain_only
        )
    else:
        url = args.url
        strategy = args.strategy
        max_depth = args.max_depth
        max_pages_per_url = args.max_pages_per_url
        max_total_nodes = args.max_total_nodes
        max_total_pages = args.max_total_pages
        same_domain_only = args.same_domain_only

    site_dir = args.sitecontent.resolve()
    persist_dir = args.persist.resolve()
    embed_model = args.embed_model

    # Step 1: Crawl
    log.info("Starting crawler...")
    run_crawler(
        url,
        strategy,
        max_depth,
        max_pages_per_url,
        max_total_nodes,
        max_total_pages,
        same_domain_only,
    )

    # Crawled output lands under sitecontent/<domain>; give that folder its own
    # collection so a second site never lands in the same namespace.
    domain = urlparse(url).netloc.replace(".", "-").replace(":", "-")
    domain_dir = site_dir / domain
    ingest_dir = domain_dir if domain_dir.exists() else site_dir
    collection = collection_for_session(session_id_for_folder(ingest_dir))

    # Step 2: Ingest
    log.info("Starting ingest...")
    run_ingest(ingest_dir, persist_dir, embed_model, args.min_chars, collection)

    # Step 3: Inspect
    log.info("Starting inspect...")
    run_inspect(persist_dir, collection)

    # Step 4: Chat (RAG)
    if args.chat:
        run_chat(
            persist_dir=persist_dir,
            collection=collection,
            embed_model=embed_model,
            top_k=args.top_k,
            provider_arg=args.llm_provider,
            model_name_arg=args.llm_model_name,
        )

    log.info("Orchestration complete!")


if __name__ == "__main__":
    main()
