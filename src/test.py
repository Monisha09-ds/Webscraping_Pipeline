# import sys
# from pathlib import Path
# import argparse

# # Path bootstrap
# PROJ_ROOT = Path(__file__).resolve().parents[1]  # Assuming main.py is in pipelines/
# SRC_DIR = PROJ_ROOT / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# # Imports from the scripts
# from utils.logger import get_logger
# from scraper.recursion import recursive_crawl
# from utils.config import CRAWL_STRATEGY_DEFAULT, SITECONTENT_ROOT, CHROMA_ROOT, EMBED_MODEL_PATH
# from vectorstore.store import build_from_sitecontent, ChromaStore
# from vectorstore.embeddings import LocalTextEmbedder

# ### NEW: import rag components
# from rag.answer import generate_answer
# from rag.llm import LLMWrapper

# log = get_logger("orchestrator")


# # ---------------- Utils ----------------
# def ask_int(prompt: str, default: int) -> int:
#     try:
#         v = input(f"{prompt} [default={default}]: ").strip()
#         return int(v) if v else default
#     except ValueError:
#         return default

# def ask_bool(prompt: str, default_yes=True) -> bool:
#     v = input(f"{prompt} [{'Y/n' if default_yes else 'y/N'}]: ").strip().lower()
#     if not v: return default_yes
#     return v in ("y","yes")

# def ask_choice(prompt: str, choices: list[str], default: str) -> str:
#     v = input(f"{prompt} {choices} [default={default}]: ").strip().lower()
#     return v if v in choices else default


# # ---------------- Core Functions ----------------
# def run_crawler(url: str, strategy: str, max_depth: int, max_pages_per_url: int, max_total_nodes: int, max_total_pages: int, same_domain_only: bool) -> int:
#     count = recursive_crawl(
#         start_url=url,
#         strategy=strategy,
#         max_depth=max_depth,
#         max_pages_per_url=max_pages_per_url,
#         max_total_pages=max_total_pages,
#         max_total_nodes=max_total_nodes,
#         same_domain_only=same_domain_only,
#     )
#     log.info(f"Crawler done. Total pages saved: {count}")
#     return count

# def run_ingest(site_dir: Path, persist_dir: Path, model_path: Path, chunk_tokens: int, chunk_overlap: int, min_chars: int):
#     if not site_dir.exists():
#         raise ValueError(f"sitecontent not found: {site_dir}")
#     if not any(site_dir.rglob("*.md")):
#         raise ValueError(f"No .md files under: {site_dir}")
#     if not model_path.exists():
#         raise ValueError(f"Embedding model path not found: {model_path}")

#     embedder = LocalTextEmbedder(model_path=str(model_path))
#     dummy_text = "Hello, world!"
#     emb = embedder.embed_documents([dummy_text])
#     log.info("Embedding model tested successfully")
#     embedder.cleanup()

#     store = build_from_sitecontent(
#         site_dir=site_dir,
#         persist_dir=persist_dir,
#         embed_model_path=str(model_path),
#         min_chars=min_chars
#     )
#     log.info(f"Ingest done. Docs persisted: {store.count()}")

# def run_inspect(persist_dir: Path):
#     store = ChromaStore(persist_dir=persist_dir)
#     log.info(f"Docs in DB: {store.count()}")
#     log.info(f"Persistence dir: {persist_dir}")

# ### NEW: query function
# def run_query(persist_dir: Path, model_path: Path, top_k: int = 3):
#     store = ChromaStore(persist_dir=persist_dir)
#     embedder = LocalTextEmbedder(model_path=str(model_path))
#     llm = LLMWrapper(mode="local", model_name="ggml-gpt4all-j-v1.3-groovy")  # Example local model

#     log.info("Query mode started. Type 'exit' to quit.")
#     while True:
#         q = input("Your Question: ").strip()
#         if q.lower() in ("exit", "quit"):
#             break
#         try:
#             answer = generate_answer(
#                 query=q,
#                 store=store,
#                 embedder=embedder,
#                 top_k=top_k,
#                 llm=llm
#             )
#             print("\n[Answer]\n" + (answer or "").strip())
#         except Exception as e:
#             print("[Error during answering:]", e)


# # ---------------- Main ----------------
# def main():
#     parser = argparse.ArgumentParser(description="Orchestrate crawler, ingest, inspect, and query pipelines")
#     parser.add_argument("--url", type=str, help="Start URL for crawling (required for full run)")
#     parser.add_argument("--strategy", type=str, default=CRAWL_STRATEGY_DEFAULT, choices=["bfs", "dfs"], help="Crawl strategy")
#     parser.add_argument("--max-depth", type=int, default=2, help="Max recursion depth")
#     parser.add_argument("--max-pages-per-url", type=int, default=5, help="Max pages per URL")
#     parser.add_argument("--max-total-nodes", type=int, default=100, help="Global node cap")
#     parser.add_argument("--max-total-pages", type=int, default=200, help="Global page cap")
#     parser.add_argument("--same-domain-only", action="store_true", default=True, help="Restrict to same domain")
#     parser.add_argument("--sitecontent", type=Path, default=SITECONTENT_ROOT, help="Sitecontent folder")
#     parser.add_argument("--persist", type=Path, default=CHROMA_ROOT, help="Chroma persist dir")
#     parser.add_argument("--embed-model", type=Path, default=EMBED_MODEL_PATH, help="Embedding model path")
#     parser.add_argument("--chunk-tokens", type=int, default=450, help="Chunk tokens")
#     parser.add_argument("--chunk-overlap", type=int, default=90, help="Chunk overlap")
#     parser.add_argument("--min-chars", type=int, default=80, help="Min chars per file")
#     parser.add_argument("--interactive", action="store_true", help="Interactive mode for inputs")
#     ### NEW
#     parser.add_argument("--query", action="store_true", help="Enter query mode after pipelines")
#     parser.add_argument("--top-k", type=int, default=3, help="Top K docs for retrieval")

#     args = parser.parse_args()

#     if args.interactive or not args.url:
#         url = input("Enter the START URL to crawl: ").strip()
#         if not url:
#             log.error("No URL provided.")
#             return
#         strategy = ask_choice("Crawl strategy", ["bfs","dfs"], args.strategy)
#         max_depth = ask_int("Max recursion depth (0 = only start page)", args.max_depth)
#         max_pages_per_url = ask_int("Max pages per URL (pagination cap)", args.max_pages_per_url)
#         max_total_nodes = ask_int("Global node cap (unique URLs)", args.max_total_nodes)
#         max_total_pages = ask_int("Global page cap (safety limit)", args.max_total_pages)
#         same_domain_only = ask_bool("Restrict to same registrable/host scope? (recomm: Y)", args.same_domain_only)
#     else:
#         url = args.url
#         strategy = args.strategy
#         max_depth = args.max_depth
#         max_pages_per_url = args.max_pages_per_url
#         max_total_nodes = args.max_total_nodes
#         max_total_pages = args.max_total_pages
#         same_domain_only = args.same_domain_only

#     site_dir = args.sitecontent.resolve()
#     persist_dir = args.persist.resolve()
#     model_path = args.embed_model.resolve()

#     # Step 1: Crawl
#     log.info("Starting crawler...")
#     run_crawler(url, strategy, max_depth, max_pages_per_url, max_total_nodes, max_total_pages, same_domain_only)

#     # Step 2: Ingest
#     log.info("Starting ingest...")
#     run_ingest(site_dir, persist_dir, model_path, args.chunk_tokens, args.chunk_overlap, args.min_chars)

#     # Step 3: Inspect
#     log.info("Starting inspect...")
#     run_inspect(persist_dir)

#     # Step 4: Query
#     if args.query:
#         run_query(persist_dir, model_path, args.top_k)

#     log.info("Orchestration complete!")


# if __name__ == "__main__":
#     main()
