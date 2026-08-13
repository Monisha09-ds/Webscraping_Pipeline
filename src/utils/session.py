# src/utils/session.py - Session identity + on-disk persistence.
#
# Sessions must survive an API restart and must be identical across the scraper
# service and the chat service, so the id is *derived* from the content folder
# rather than randomly generated.

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from utils.config import SESSIONS_ROOT

logger = logging.getLogger(__name__)

# Fixed namespace so every process derives the same id for the same folder.
_SESSION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "webscraper-pipeline/session")

# Only these keys are JSON-serializable; the rest (llm, store, embedder) are
# live objects rebuilt on demand.
_PERSISTED_KEYS = ("folder", "collection_name", "chat_history", "embed_model")


def session_id_for_folder(folder: Path | str) -> str:
    """
    Stable session id for a scraped-content folder.

    The same folder always yields the same id, in any process, on any host, so
    the scraper API, the chat API and both Streamlit UIs agree without having to
    pass ids around.
    """
    key = Path(folder).expanduser().resolve().as_posix().rstrip("/").lower()
    return str(uuid.uuid5(_SESSION_NAMESPACE, key))


def _session_file(session_id: str) -> Path:
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    return SESSIONS_ROOT / f"{session_id}.json"


def load_session(session_id: str) -> dict | None:
    """Load persisted session data, or None if it was never saved."""
    path = _session_file(session_id)
    if not path.exists():
        logger.debug("Session %s not found at %s", session_id, path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Session %s is unreadable (%s); ignoring.", session_id, e)
        return None
    logger.info("Session %s loaded from %s", session_id, path)
    return data


def save_session(session_id: str, data: dict[str, Any]) -> None:
    """Persist the serializable subset of a session to disk."""
    path = _session_file(session_id)
    payload = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in data.items()
        if k in _PERSISTED_KEYS
    }
    payload["session_id"] = session_id
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Session %s saved to %s", session_id, path)
    except OSError as e:
        logger.warning("Could not save session %s: %s", session_id, e)


def list_sessions() -> list[dict]:
    """Every persisted session, newest first."""
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(SESSIONS_ROOT.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out
