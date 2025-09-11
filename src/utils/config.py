
####-----------------####
from __future__ import annotations
import sys
from pathlib import Path

# --- Project root bootstrap (Rule #1) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Paths ---
SITECONTENT_DIR = PROJECT_ROOT / "sitecontent"
SITECONTENT_DIR.mkdir(parents=True, exist_ok=True)

# --- Crawler knobs ---
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
PAGE_TIMEOUT_MS = 120000
CACHE_MODE = "BYPASS"

# --- Crawl strategy ---
CRAWL_STRATEGY_DEFAULT = "bfs"   # or "dfs"

# --- Pagination ---
MAX_PAGES_DEFAULT = 25
FOLLOW_REL_NEXT = True
NEXT_TEXT_HINTS = ("next", "older", "more", ">>", "›", "→")

# --- Recursion defaults ---
MAX_DEPTH_DEFAULT = 3
MAX_TOTAL_PAGES_DEFAULT = 5
SAME_DOMAIN_ONLY_DEFAULT = True
POLITE_DELAY_SEC = 0.5
SCOPE_MODE = "host"
ALLOWLIST_DOMAINS = set()
PATH_PREFIXES = []

# --- Global crawl caps ---
MAX_TOTAL_NODES_DEFAULT = 100
MAX_TOTAL_PAGES_DEFAULT = 200

# --- Prioritization ---
PRIORITIZE_SAME_HOST = True
PRIORITIZE_SHALLOW_PATHS = True
MAX_OUTGOING_PER_PAGE = 50

# --- URL normalization / dedup ---
STRIP_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src", "mkt_tok"
}

# --- Save/skip policy ---
REQUIRE_TITLE_META = True
REQUIRE_NONEMPTY_MD = False

SKIP_EXTS = {
    ".pdf",".jpg",".jpeg",".png",".gif",".webp",".svg",
    ".zip",".gz",".tar",".tgz",".rar",".7z",".mp3",".mp4",".mov",".avi",".mkv",
    ".css",".js",".ico",".woff",".woff2",".ttf",".eot"
}

# --- Redirect handling ---
PRE_RESOLVE_REDIRECTS = True
STRICT_CRAWL4AI_ONLY = True
WRITE_REDIRECT_STUB = True


PROJ_ROOT = Path(__file__).resolve().parents[3]      # website_content_scrapper
APP_ROOT  = PROJ_ROOT / "webscraper_pipeline"        # webscraper_pipeline
SRC_ROOT  = APP_ROOT / "src"

SITECONTENT_ROOT = APP_ROOT / "sitecontent"          # crawled markdown
OUTPUT_ROOT      = APP_ROOT / "output"               # app outputs
CHROMA_ROOT      = APP_ROOT / "chroma_store"         # chroma persistence

# --- Collection name (use everywhere!) ---
COLLECTION_NAME = "rag_docs"

# --- Local models ---
MODELS_ROOT      = APP_ROOT / "models"
EMBED_MODEL_PATH = str(MODELS_ROOT / "colnomic-embed-multimodal-3b")
LLM_MODEL_PATH   = str(MODELS_ROOT / "gemma-3-4b-it")


def ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHROMA_ROOT.mkdir(parents=True, exist_ok=True)


if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
