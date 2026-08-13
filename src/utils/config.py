####-----------------####
# src/utils/config.py
# Single source of truth for paths, crawler knobs, and model selection.
# Everything that varies per-environment is read from the environment (.env).

from __future__ import annotations
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# --- Project root bootstrap ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from the project root regardless of the current working directory.
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default


# ============================================================
# Paths
# ============================================================
APP_ROOT = PROJECT_ROOT
SRC_ROOT = APP_ROOT / "src"

SITECONTENT_ROOT = Path(os.getenv("SITECONTENT_ROOT") or (APP_ROOT / "sitecontent"))
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT") or (APP_ROOT / "output"))
CHROMA_ROOT = Path(os.getenv("CHROMA_ROOT") or (APP_ROOT / "chroma_store"))
SESSIONS_ROOT = Path(os.getenv("SESSIONS_ROOT") or (APP_ROOT / "sessions"))

# Backwards-compatible alias (older modules imported SITECONTENT_DIR).
SITECONTENT_DIR = SITECONTENT_ROOT


def ensure_dirs() -> None:
    """Create every runtime directory the app writes to."""
    for d in (SITECONTENT_ROOT, OUTPUT_ROOT, CHROMA_ROOT, SESSIONS_ROOT):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

# ============================================================
# Embeddings
# ============================================================
# Any sentence-transformers model id or a local directory path.
#   nomic-ai/nomic-embed-text-v1.5  -> 0.1B params,  768 dims, 8192 ctx  (default)
#   BAAI/bge-m3                     -> 0.57B params, 1024 dims, multilingual
#   Qwen/Qwen3-Embedding-0.6B       -> 0.6B params,  1024 dims
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto")  # auto | cpu | cuda
EMBED_BATCH_SIZE = _env_int("EMBED_BATCH_SIZE", 16)
EMBED_MAX_SEQ_LENGTH = _env_int("EMBED_MAX_SEQ_LENGTH", 1024)
EMBED_TRUST_REMOTE_CODE = _env_bool("EMBED_TRUST_REMOTE_CODE", True)

# Nomic models are asymmetric: documents and queries need different prefixes.
# Symmetric models (bge-m3, qwen3) use empty prefixes.
_DEFAULT_PREFIXES = {
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1": ("search_document: ", "search_query: "),
}
_doc_prefix, _query_prefix = _DEFAULT_PREFIXES.get(EMBED_MODEL_NAME, ("", ""))
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", _doc_prefix)
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", _query_prefix)

# Kept so older call sites that passed a "path" keep working.
EMBED_MODEL_PATH = EMBED_MODEL_NAME

# ============================================================
# LLM
# ============================================================
# groq  -> open-source models served by Groq (llama, gpt-oss, qwen ...)
# local -> a HuggingFace causal LM from a local folder (offline fallback)
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()

# Model ids Groq serves in production. Preview models work too; they are not
# listed here because Groq rotates them.
GROQ_MODEL_CHOICES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1024)

MODELS_ROOT = Path(os.getenv("MODELS_ROOT") or (APP_ROOT / "models"))
LOCAL_LLM_PATH = Path(os.getenv("LOCAL_LLM_PATH") or (MODELS_ROOT / "gemma-3-4b-it"))

# ============================================================
# Vector store
# ============================================================
# Per-session collections are named f"{COLLECTION_PREFIX}_{session_id}" so two
# scraped sites never share a namespace.
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "site")
COLLECTION_NAME = COLLECTION_PREFIX  # legacy alias

RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 5)
INGEST_MIN_CHARS = _env_int("INGEST_MIN_CHARS", 80)


def collection_for_session(session_id: str) -> str:
    """Chroma collection name owned by a single scraped site / session."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(session_id))
    return f"{COLLECTION_PREFIX}_{safe}"


# ============================================================
# Service URLs (frontends -> backends)
# ============================================================
SCRAPER_API_URL = os.getenv("SCRAPER_API_URL", "http://localhost:5005")
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:5003")
SCRAPER_API_PORT = _env_int("SCRAPER_API_PORT", 5005)
CHAT_API_PORT = _env_int("CHAT_API_PORT", 5003)

# Comma-separated list of origins allowed to call the APIs.
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,"
        "http://localhost:8502,http://127.0.0.1:8502",
    ).split(",")
    if o.strip()
]

# ============================================================
# Crawler knobs
# ============================================================
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)
PAGE_TIMEOUT_MS = _env_int("PAGE_TIMEOUT_MS", 120000)
CACHE_MODE = os.getenv("CACHE_MODE", "BYPASS")

# --- Crawl strategy ---
CRAWL_STRATEGY_DEFAULT = os.getenv("CRAWL_STRATEGY_DEFAULT", "bfs")  # bfs | dfs

# --- Pagination ---
MAX_PAGES_DEFAULT = _env_int("MAX_PAGES_DEFAULT", 25)
FOLLOW_REL_NEXT = _env_bool("FOLLOW_REL_NEXT", True)
NEXT_TEXT_HINTS = ("next", "older", "more", ">>", "›", "→")

# --- Recursion defaults ---
MAX_DEPTH_DEFAULT = _env_int("MAX_DEPTH_DEFAULT", 3)
SAME_DOMAIN_ONLY_DEFAULT = _env_bool("SAME_DOMAIN_ONLY_DEFAULT", True)
POLITE_DELAY_SEC = _env_float("POLITE_DELAY_SEC", 0.5)
SCOPE_MODE = os.getenv("SCOPE_MODE", "host")
ALLOWLIST_DOMAINS: set[str] = set()
PATH_PREFIXES: list[str] = []

# --- Global crawl caps ---
MAX_TOTAL_NODES_DEFAULT = _env_int("MAX_TOTAL_NODES_DEFAULT", 100)
MAX_TOTAL_PAGES_DEFAULT = _env_int("MAX_TOTAL_PAGES_DEFAULT", 200)

# --- Prioritization ---
PRIORITIZE_SAME_HOST = True
PRIORITIZE_SHALLOW_PATHS = True
MAX_OUTGOING_PER_PAGE = _env_int("MAX_OUTGOING_PER_PAGE", 50)

# --- URL normalization / dedup ---
STRIP_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src", "mkt_tok",
}

# --- Save/skip policy ---
REQUIRE_TITLE_META = _env_bool("REQUIRE_TITLE_META", True)
REQUIRE_NONEMPTY_MD = _env_bool("REQUIRE_NONEMPTY_MD", False)

SKIP_EXTS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".zip", ".gz", ".tar", ".tgz", ".rar", ".7z", ".mp3", ".mp4", ".mov", ".avi", ".mkv",
    ".css", ".js", ".ico", ".woff", ".woff2", ".ttf", ".eot",
}

# --- Redirect handling ---
PRE_RESOLVE_REDIRECTS = True
STRICT_CRAWL4AI_ONLY = True
WRITE_REDIRECT_STUB = True


if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
