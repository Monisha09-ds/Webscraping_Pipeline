####-------------####

from __future__ import annotations
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import re
from typing import Optional, Dict, Any, Iterable, Tuple
from utils.config import NEXT_TEXT_HINTS, FOLLOW_REL_NEXT

def _bump_query_page(url: str, next_page_number: int) -> Optional[str]:
    """Handle URLs like ?page=1 or ?p=1."""
    p = urlparse(url)
    q = parse_qs(p.query, keep_blank_values=True)
    key = None
    for candidate in ("page", "p"):
        if candidate in q:
            key = candidate
            break
    if not key:
        return None
    q[key] = [str(next_page_number)]
    new_query = urlencode(q, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))

def _bump_trailing_page_segment(url: str, next_page_number: int) -> Optional[str]:
    """
    Handle .../page/2 or .../p/2 patterns.
    """
    p = urlparse(url)
    path = p.path
    m = re.search(r"(?:/page|/p)/(\d+)(/?)$", path)
    if not m:
        return None
    new_path = re.sub(r"(?:/page|/p)/\d+/?$", f"/page/{next_page_number}", path)
    return urlunparse((p.scheme, p.netloc, new_path, p.params, p.query, p.fragment))

def infer_next_url_from_pattern(current_url: str, current_index: int) -> Optional[str]:
    """
    Only used if we don't find a 'rel=next' or anchor 'Next' link.
    """
    n = current_index + 1
    q = _bump_query_page(current_url, n)
    if q:
        return q
    s = _bump_trailing_page_segment(current_url, n)
    if s:
        return s
    return None

def pick_next_from_links(base_url: str, links: Dict[str, Any]) -> Optional[str]:
    """
    Use Crawl4AI's result.links to find 'next' candidates.
    """
    all_links: Iterable[str] = []
    for key in ("internal", "external", "other"):
        v = links.get(key)
        if isinstance(v, list):
            all_links += v

    lowered = [l for l in all_links if isinstance(l, str)]
    for href in lowered:
        txt = href.lower()
        if any(h in txt for h in NEXT_TEXT_HINTS):
            return urljoin(base_url, href)
    return None


