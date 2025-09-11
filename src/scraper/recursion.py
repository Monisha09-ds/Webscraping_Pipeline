####----------Recursive driver: pass the node cap; short-circuit on both caps--------####

from __future__ import annotations
from typing import List, Dict, Any
from utils.logger import get_logger
from utils.config import (
    MAX_PAGES_DEFAULT, MAX_DEPTH_DEFAULT,
    MAX_TOTAL_PAGES_DEFAULT, MAX_TOTAL_NODES_DEFAULT,
    SAME_DOMAIN_ONLY_DEFAULT,
)
from scraper.scraper import crawl_single_node_with_pagination
from scraper.frontier import Frontier
from scraper.markdown_saver import root_dir_for_url, child_dir
from shutil import rmtree

log = get_logger("recursive")

def _collect_links(meta: Dict[str, Any]) -> List[str]:
    links = meta.get("links", {}) or {}
    out = []
    for k in ("internal","external","other"):
        v = links.get(k)
        if isinstance(v, list): out.extend([x for x in v if isinstance(x, str)])
    return out

def recursive_crawl(
    start_url: str,
    strategy: str = "dfs", 
    max_depth: int = MAX_DEPTH_DEFAULT,
    max_pages_per_url: int = MAX_PAGES_DEFAULT,
    max_total_pages: int = MAX_TOTAL_PAGES_DEFAULT,
    max_total_nodes: int = MAX_TOTAL_NODES_DEFAULT,
    same_domain_only: bool = SAME_DOMAIN_ONLY_DEFAULT,
) -> int:
    total_pages_saved = 0
    start_dir = root_dir_for_url(start_url)

    frontier = Frontier(
        start_url=start_url,
        start_dir=start_dir,
        same_domain_only=same_domain_only,
        node_cap=max_total_nodes,   
        strategy=strategy,  
    )
    frontier.push_start()



    while True:
        item = frontier.pop()
        if not item:
            break
        url, depth, target_dir = item
        if depth > max_depth:
            continue

        log.info(f"[depth={depth}] Crawling {url}")
       
        try:
            saved_paths, last_meta, final_url = crawl_single_node_with_pagination(
                original_url=url,
                target_dir=target_dir,
                depth=depth,
                parent_url=None,
                max_pages=max_pages_per_url,
                wipe_old=True,
            )

            if not saved_paths:
    
                try:
                    if target_dir.exists() and not any(target_dir.rglob("*")):
                        rmtree(target_dir)
                except Exception:
                    pass
                continue  

            total_pages_saved += len(saved_paths)
            if total_pages_saved >= max_total_pages:
                log.info("Global **page** cap reached.")
                break

            if last_meta:
                outgoing = _collect_links(last_meta)
                def _child_builder(child_url: str, idx: int):
                    return child_dir(target_dir, child_url, idx)
                frontier.push_links(base_url=final_url, depth=depth+1, links=outgoing, child_dir_builder=_child_builder)

        except Exception as e:
            log.warning(f"Failed {url}: {e}")
            continue

        if total_pages_saved >= max_total_pages:
            log.info("Global **page** cap reached.")
            break

    log.info(f"Recursive crawl saved {total_pages_saved} page files (nodes≤{max_total_nodes}).")
    return total_pages_saved



