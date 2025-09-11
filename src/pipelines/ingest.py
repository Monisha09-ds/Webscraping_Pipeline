
####-----------------####

"""
Ingest pipeline: Markdown -> Chunks -> Local Embeddings -> ChromaDB (persistent)
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse
from rich import print
from rich.progress import track
import torch
import numpy as np

# ---- Path bootstrap ------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[3]     
SRC_DIR   = PROJ_ROOT / "webscraper_pipeline" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# -------------------------------------------------------------------------

from utils.config import SITECONTENT_ROOT, CHROMA_ROOT, EMBED_MODEL_PATH, ensure_dirs
from vectorstore.store import build_from_sitecontent
from vectorstore.embeddings import LocalTextEmbedder



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a ChromaDB index from crawled Markdown using a LOCAL embedding model."
    )
    ap.add_argument("--sitecontent", type=Path, default=SITECONTENT_ROOT,
                    help="Folder with markdown pages (default from config.py).")
    ap.add_argument("--persist", type=Path, default=CHROMA_ROOT,
                    help="ChromaDB persistent directory (default from config.py).")
    ap.add_argument("--embed-model", type=str, default=str(EMBED_MODEL_PATH),
                    help="Local HF/SBERT model directory for embeddings.")
    ap.add_argument("--chunk-tokens", type=int, default=450,
                    help="Approx tokens per chunk.")
    ap.add_argument("--chunk-overlap", type=int, default=90,
                    help="Token overlap between chunks.")
    ap.add_argument("--min-chars", type=int, default=80,
                    help="Skip markdown files with fewer chars.")
    return ap.parse_args()


def sanity_checks(site_dir: Path, model_path: Path):
    if not site_dir.exists():
        raise SystemExit(f"[red]sitecontent not found:[/red] {site_dir}")
    if not any(site_dir.rglob("*.md")):
        raise SystemExit(f"[red]No .md files under:[/red] {site_dir}\nDid you run the crawler already?")
    if not model_path.exists():
        raise SystemExit(f"[red]Embedding model path not found:[/red] {model_path}")


def test_model_load(model_path: Path):
    """Ensure embedding model loads and can generate embeddings."""
    print(f"[bold cyan]Testing embedding model load...[/bold cyan]")
    embedder = LocalTextEmbedder(model_path=str(model_path))
    dummy_text = "Hello, world!"
    emb = embedder.embed_documents([dummy_text])
    print(f"[bold green]Success:[/bold green] Model loaded and generated embedding!")
    embedder.cleanup()


def main():
    args = parse_args()
    ensure_dirs()

    site_dir = args.sitecontent.resolve()
    persist_dir = args.persist.resolve()
    model_path = Path(args.embed_model).resolve()

    print(f"[bold cyan]Ingest starting[/bold cyan]")
    print(f"  • sitecontent : [green]{site_dir}[/green]")
    print(f"  • persist dir : [green]{persist_dir}[/green]")
    print(f"  • embed model : [green]{model_path}[/green]")
    print(f"  • chunk size  : {args.chunk_tokens} tokens, overlap {args.chunk_overlap}")
    print(f"  • min chars   : {args.min_chars}")

    # 1️⃣ Sanity checks
    sanity_checks(site_dir, model_path)

    # 2️⃣ Test embedding model
    test_model_load(model_path)

    # 3️⃣ Build Chroma store & process markdown
    print("[bold cyan]Processing markdown files and creating embeddings...[/bold cyan]")
    store = build_from_sitecontent(
        site_dir=site_dir,
        persist_dir=persist_dir,
        embed_model_path=str(model_path),
        min_chars=args.min_chars
    )
    
    # 4️⃣ Show embedding stats
    stats_path = persist_dir / "stats.json"
    if stats_path.exists():
        import json
        stats = json.loads(stats_path.read_text())
        print(f"[bold green]Embeddings created:[/bold green] {stats['count']} chunks")
        print(f"[bold green]Embedding dimension:[/bold green] {stats['dim']}")

    print(f"[bold green]✅ ChromaDB persisted at:[/bold green] [yellow]{persist_dir}[/yellow]")


    print("Docs persisted:", store.col.count())

if __name__ == "__main__":
    main()
