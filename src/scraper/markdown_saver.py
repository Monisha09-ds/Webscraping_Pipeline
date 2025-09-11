####------------(hierarchical saving + meta + stubs)----------------####
# src/scraper/markdown_saver.py
import json, hashlib
from pathlib import Path
from urllib.parse import urlparse
from slugify import slugify
from utils.config import SITECONTENT_DIR, WRITE_REDIRECT_STUB
from utils.logger import get_logger

log = get_logger("saver")

def slug_host(url: str) -> str:
    return slugify(urlparse(url).netloc or "unknown-host")

def slug_path(url: str) -> str:
    p = urlparse(url)
    base = slugify((p.path or "/").strip("/") or "home")
    if p.query:
        qh = hashlib.sha1(p.query.encode("utf-8")).hexdigest()[:8]
        base = f"{base}-q{qh}"
    return base

def root_dir_for_url(url: str) -> Path:
    return SITECONTENT_DIR / slug_host(url) / slug_path(url)

def ensure_children_dir(parent_dir: Path) -> Path:
    d = parent_dir / "children"
    d.mkdir(parents=True, exist_ok=True)
    return d

def child_dir(parent_dir: Path, child_url: str, index: int) -> Path:
    """Return the *intended* child dir path (do not create it here)."""
    
    p = urlparse(child_url)
    host = slugify(p.netloc or "unknown-host")
    path_only = slug_path(child_url)  
    base_name = f"{index:03d}-{path_only}"

    children_root = parent_dir / "children"
    final_name = base_name
    if (children_root / final_name).exists():
        final_name = f"{base_name}--{host}"
    return children_root / final_name

def wipe_pages(target_dir: Path):
    if not target_dir.exists(): return
    for f in target_dir.glob("page-*.md"):
        try: f.unlink()
        except Exception as e: log.warning(f"Could not remove {f}: {e}")

def write_page(target_dir: Path, page_num: int, markdown: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    fp = target_dir / f"page-{page_num}.md"
    fp.write_text(markdown or "", encoding="utf-8")
    log.info(f"Saved {fp}")
    return fp

def write_meta(target_dir: Path, meta: dict):
    (target_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def record_redirect(original_url: str, final_url: str, original_dir: Path):
    redirects = SITECONTENT_DIR / "redirects.jsonl"
    with redirects.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"from": original_url, "to": final_url}) + "\n")

    if WRITE_REDIRECT_STUB and original_url != final_url:
        original_dir.mkdir(parents=True, exist_ok=True)
        (original_dir / "page-1.md").write_text(
            f"# Redirected\n\nThis URL redirects to:\n{final_url}\n",
            encoding="utf-8"
        )
