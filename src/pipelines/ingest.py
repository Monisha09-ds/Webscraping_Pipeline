
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
PROJ_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# -------------------------------------------------------------------------

from utils.config import (
    SITECONTENT_ROOT,
    CHROMA_ROOT,
    COLLECTION_PREFIX,
    EMBED_MODEL_NAME,
    ensure_dirs,
    collection_for_session,
)
from utils.session import session_id_for_folder
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
    ap.add_argument("--embed-model", type=str, default=EMBED_MODEL_NAME,
                    help="Embedding model id (HuggingFace) or local model directory.")
    ap.add_argument("--collection", type=str, default=None,
                    help="Chroma collection name. Defaults to one derived from --sitecontent.")
    ap.add_argument("--min-chars", type=int, default=80,
                    help="Skip markdown files with fewer chars.")
    return ap.parse_args()


def sanity_checks(site_dir: Path):
    if not site_dir.exists():
        raise SystemExit(f"[red]sitecontent not found:[/red] {site_dir}")
    if not any(site_dir.rglob("*.md")):
        raise SystemExit(f"[red]No .md files under:[/red] {site_dir}\nDid you run the crawler already?")


def main():
    args = parse_args()
    ensure_dirs()

    site_dir = args.sitecontent.resolve()
    persist_dir = args.persist.resolve()
    embed_model = args.embed_model
    collection = args.collection or collection_for_session(session_id_for_folder(site_dir))

    print("[bold cyan]Ingest starting[/bold cyan]")
    print(f"  • sitecontent : [green]{site_dir}[/green]")
    print(f"  • persist dir : [green]{persist_dir}[/green]")
    print(f"  • embed model : [green]{embed_model}[/green]")
    print(f"  • collection  : [green]{collection}[/green]")
    print(f"  • min chars   : {args.min_chars}")

    sanity_checks(site_dir)

    print("[bold cyan]Processing markdown files and creating embeddings...[/bold cyan]")
    store = build_from_sitecontent(
        site_dir=site_dir,
        persist_dir=persist_dir,
        embed_model_path=embed_model,
        min_chars=args.min_chars,
        collection=collection,
    )

    stats_path = persist_dir / f"stats_{collection}.json"
    if stats_path.exists():
        import json
        stats = json.loads(stats_path.read_text())
        print(f"[bold green]Embeddings created:[/bold green] {stats['count']} chunks")
        print(f"[bold green]Embedding dimension:[/bold green] {stats['dim']}")

    print(f"[bold green]✅ ChromaDB persisted at:[/bold green] [yellow]{persist_dir}[/yellow]")
    print("Chunks persisted:", store.count())

if __name__ == "__main__":
    main()
