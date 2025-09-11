
####---------(save into provided target_dir; write _meta.json)------------####
# src/scraper/scraper.py
from __future__ import annotations
import sys, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.config import FOLLOW_REL_NEXT, POLITE_DELAY_SEC, MAX_PAGES_DEFAULT
from scraper.extractor import fetch_markdown
from scraper.pagination import pick_next_from_links, infer_next_url_from_pattern
from scraper.markdown_saver import write_page, wipe_pages, write_meta, record_redirect
from utils.config import FOLLOW_REL_NEXT, POLITE_DELAY_SEC, MAX_PAGES_DEFAULT, REQUIRE_TITLE_META, REQUIRE_NONEMPTY_MD

log = get_logger("scraper")

def crawl_single_node_with_pagination(
    original_url: str,
    target_dir: Path,
    depth: int,
    parent_url: Optional[str],
    max_pages: int = MAX_PAGES_DEFAULT,
    wipe_old: bool = True
) -> Tuple[List[str], Optional[Dict[str, Any]], str]:
    """
    Crawl original_url, save into target_dir (hierarchical layout).
    Returns (saved_paths, last_meta, final_url).
    """
    if wipe_old: wipe_pages(target_dir)

    saved: List[str] = []
    page_num = 1
    cur_url = original_url
    last_meta: Optional[Dict[str, Any]] = None
    visited_chain = set()
    final_first_url: Optional[str] = None

    while cur_url and page_num <= max_pages:
        if cur_url in visited_chain: break
        visited_chain.add(cur_url)

        md, meta, final_url = fetch_markdown(cur_url)

        # ---- NEW: skip policy on page-1 only ----
        if page_num == 1:
            title_ok = bool((meta or {}).get("title"))
            body_ok  = bool((md or "").strip())
            if (REQUIRE_TITLE_META and not title_ok) or (REQUIRE_NONEMPTY_MD and not body_ok):
                # Don't save anything for this node; signal "skipped"
                return [], None, final_url
            
        if final_first_url is None:
            final_first_url = final_url
            if final_first_url != original_url:
                record_redirect(original_url, final_first_url, target_dir)

        fp = write_page(target_dir, page_num, md)
        saved.append(str(fp))
        last_meta = meta

        # next page detection
        next_url = None
        if FOLLOW_REL_NEXT:
            next_url = pick_next_from_links(final_url, meta.get("links", {}))
        if not next_url:
            next_url = infer_next_url_from_pattern(final_url, page_num)
        if not next_url or next_url == final_url:
            break

        cur_url = next_url
        page_num += 1
        time.sleep(POLITE_DELAY_SEC)

    write_meta(target_dir, {
        "url": original_url,
        "final_url": final_first_url or original_url,
        "parent_url": parent_url,
        "depth": depth,
        "title": (last_meta or {}).get("title"),
        "saved_pages": len(saved),
    })
    return saved, last_meta, (final_first_url or original_url)
