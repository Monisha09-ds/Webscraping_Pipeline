# ###------(interactive prompts; unchanged except mode)------------###

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[3]      
SRC_DIR   = PROJ_ROOT / "webscraper_pipeline" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.logger import get_logger
from scraper.recursion import recursive_crawl
from utils.config import CRAWL_STRATEGY_DEFAULT

log = get_logger("main")

def _ask_int(prompt: str, default: int) -> int:
    try:
        v = input(f"{prompt} [default={default}]: ").strip()
        return int(v) if v else default
    except ValueError:
        return default

def _ask_bool(prompt: str, default_yes=True) -> bool:
    v = input(f"{prompt} [{'Y/n' if default_yes else 'y/N'}]: ").strip().lower()
    if not v: return default_yes
    return v in ("y","yes")


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    v = input(f"{prompt} {choices} [default={default}]: ").strip().lower()
    return v if v in choices else default

def main():
    url = input("Enter the START URL to crawl: ").strip()
    if not url:
        log.error("No URL provided."); return

    strategy = _ask_choice("Crawl strategy", ["bfs","dfs"], CRAWL_STRATEGY_DEFAULT)  # NEW
    max_depth = _ask_int("Max recursion depth (0 = only start page)", 2)
    max_pages_per_url = _ask_int("Max pages per URL (pagination cap)", 5)
    max_total_nodes = _ask_int("Global node cap (unique URLs)", 100)
    max_total_pages = _ask_int("Global page cap (safety limit)", 200)
    same_domain_only = _ask_bool("Restrict to same registrable/host scope? (recomm: Y)", True)

    count = recursive_crawl(
        start_url=url,
        strategy=strategy,                         # NEW
        max_depth=max_depth,
        max_pages_per_url=max_pages_per_url,
        max_total_pages=max_total_pages,
        max_total_nodes=max_total_nodes,
        same_domain_only=same_domain_only,
    )
    log.info(f"Done. Total pages saved: {count}")


if __name__ == "__main__":
    main()