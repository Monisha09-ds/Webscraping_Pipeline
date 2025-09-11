# utils/session.py - Session persistence helpers

from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path("/home/aiteam/Desktop/website_content_scrapper/webscraper_pipeline/sessions")  # Adjust to your path
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def load_session(session_id: str) -> dict | None:
    """Load session data from JSON if exists."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Session {session_id} loaded from {session_file}")
        return data
    logger.warning(f"Session {session_id} not found at {session_file}")
    return None

def save_session(session_id: str, data: dict):
    """Save serializable session data to JSON."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    serializable_data = {k: str(v) if isinstance(v, Path) else v for k, v in data.items() if k in ["folder", "collection_name", "chat_history"]}  # Serialize Path and keep serializable keys
    with open(session_file, 'w') as f:
        json.dump(serializable_data, f)
    logger.info(f"Session {session_id} saved to {session_file}")