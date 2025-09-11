## ------Page Iteration -with (BFS?DFS)------###

from __future__ import annotations
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag, parse_qsl, urlencode, urlunparse
from typing import Iterable, Deque, Optional, Tuple, Set, List
import tldextract
from utils.config import (
    SKIP_EXTS, SCOPE_MODE, ALLOWLIST_DOMAINS, PATH_PREFIXES,
    PRIORITIZE_SAME_HOST, PRIORITIZE_SHALLOW_PATHS, STRIP_TRACKING_PARAMS,
    MAX_OUTGOING_PER_PAGE, MAX_TOTAL_NODES_DEFAULT,
)

def _strip_tracking(u: str) -> str:
    p = urlparse(u)
    q = [(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in STRIP_TRACKING_PARAMS]
    return urlunparse((p.scheme, p.netloc, p.path.rstrip("/") or "/", p.params, urlencode(q, doseq=True), ""))

def normalize(base_url: str, href: str) -> Optional[str]:
    if not href: return None
    u = urljoin(base_url, href)
    u, _ = urldefrag(u)
    p = urlparse(u)
    if p.scheme not in ("http","https"): return None
    path = (p.path or "").lower()
    if any(path.endswith(ext) for ext in SKIP_EXTS): return None
    u = _strip_tracking(u)
    if u.endswith("/index.html"):
        u = u[: -len("index.html")]
    return u

def same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()

def same_reg_domain(a: str, b: str) -> bool:
    e1, e2 = tldextract.extract(a), tldextract.extract(b)
    return (e1.registered_domain or a) == (e2.registered_domain or b)

def path_allowed(u: str) -> bool:
    if not PATH_PREFIXES: return True
    path = urlparse(u).path or "/"
    return any(path.startswith(p) for p in PATH_PREFIXES)

def _score(base_url: str, candidate: str) -> tuple:
    host_same = 0 if same_host(base_url, candidate) else 1
    depth = (urlparse(candidate).path or "/").strip("/").count("/")
    length = len(candidate)
    return (host_same if PRIORITIZE_SAME_HOST else 0,
            depth if PRIORITIZE_SHALLOW_PATHS else 0,
            length)

class Frontier:
    """
    Container holds (url, depth, target_dir). 
    BFS => FIFO (pop left). DFS => LIFO (pop right).
    """
    def __init__(self, start_url: str, start_dir, same_domain_only: bool,
                 node_cap: int = MAX_TOTAL_NODES_DEFAULT,
                 strategy: str = "bfs"):
        self.start_url = start_url
        self.start_dir = start_dir
        self.same_domain_only = same_domain_only
        self._seen: Set[str] = set()
        self._q: Deque[Tuple[str, int, object]] = deque()
        self._node_cap = max(1, node_cap)
        self._visited_nodes = 0
        self._strategy = strategy if strategy in ("bfs","dfs") else "bfs"

    def push_start(self):
        self._q.append((self.start_url, 0, self.start_dir))

    def push_links(self, base_url: str, depth: int, links: Iterable[str], child_dir_builder):
        candidates: List[str] = []
        for href in links:
            u = normalize(base_url, href)
            if not u: continue
            if not path_allowed(u): continue

            ok = True
            if SCOPE_MODE == "host":
                ok = same_host(self.start_url, u)
            elif SCOPE_MODE == "registrable":
                ok = same_reg_domain(self.start_url, u)
            elif SCOPE_MODE == "allowlist":
                ok = urlparse(u).netloc.lower() in {d.lower() for d in ALLOWLIST_DOMAINS}
            if self.same_domain_only and not ok:
                continue
            if u in self._seen:
                continue
            candidates.append(u)

        # rank by priority
        candidates = sorted(set(candidates), key=lambda u: _score(self.start_url, u))

        # IMPORTANT: for DFS we want the highest-priority URL to be popped first.
        # Since DFS pops from the RIGHT, we append in REVERSE order.
        if self._strategy == "dfs":
            iterable = list(reversed(candidates))
        else:  # bfs
            iterable = candidates

        limit = min(MAX_OUTGOING_PER_PAGE or len(iterable), len(iterable))
        for idx, u in enumerate(iterable[:limit], start=1):
            child_dir = child_dir_builder(u, idx)
            self._q.append((u, depth, child_dir))

    def pop(self) -> Optional[Tuple[str, int, object]]:
        if self._visited_nodes >= self._node_cap:
            return None
        while self._q:
            if self._strategy == "dfs":
                u, d, tdir = self._q.pop()     # LIFO
            else:
                u, d, tdir = self._q.popleft() # FIFO
            if u in self._seen:
                continue
            if self._visited_nodes >= self._node_cap:
                return None
            self._seen.add(u)
            self._visited_nodes += 1
            return u, d, tdir
        return None

