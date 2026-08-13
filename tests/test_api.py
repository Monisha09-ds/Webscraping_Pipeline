from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.chat_api import app as chat_app
from api.scraper_endpoints import app as scraper_app

scraper_client = TestClient(scraper_app)
chat_client = TestClient(chat_app)


# ---------------- Scraper API ----------------
def test_scraper_root():
    resp = scraper_client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Scraper API is alive"}


def test_scraper_health():
    resp = scraper_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_scrape_rejects_bad_url():
    resp = scraper_client.post("/scrape", json={"url": "not-a-url"})
    assert resp.status_code == 422


@patch("api.scraper_endpoints._run_crawler_sync")
def test_scrape_endpoint(mock_crawl, tmp_path):
    mock_crawl.return_value = 5

    site_root = tmp_path / "sitecontent"
    (site_root / "example-com" / "home").mkdir(parents=True)
    (site_root / "example-com" / "home" / "page-1.md").write_text("# hi", encoding="utf-8")

    with patch("api.scraper_endpoints.SITECONTENT_ROOT", site_root), \
         patch("api.scraper_endpoints.save_session"):
        resp = scraper_client.post("/scrape", json={"url": "http://example.com"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["pages_saved"] == 5
    assert body["files_found"] == 1
    assert body["session_id"]
    assert body["collection_name"].startswith("site_")


@patch("api.scraper_endpoints._run_crawler_sync")
def test_scrape_reports_zero_pages(mock_crawl, tmp_path):
    mock_crawl.return_value = 0
    site_root = tmp_path / "sitecontent"
    site_root.mkdir()

    with patch("api.scraper_endpoints.SITECONTENT_ROOT", site_root):
        resp = scraper_client.post("/scrape", json={"url": "http://example.com"})

    assert resp.status_code == 422
    assert "No pages were scraped" in resp.json()["detail"]


# ---------------- Chat API ----------------
def test_chat_root():
    resp = chat_client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Web Scraper RAG Pipeline API"}


def test_chat_health_reports_models():
    resp = chat_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "embed_model" in body
    assert "llama-3.3-70b-versatile" in body["llm_model_choices"]


def test_unknown_session_is_404():
    resp = chat_client.post("/chat", json={"session_id": "nope", "message": "hi"})
    assert resp.status_code == 404


def test_check_session_is_deterministic(tmp_path):
    """The same folder always maps to the same session id."""
    with patch("api.chat_api.collection_exists", return_value=False):
        a = chat_client.get("/check_session", params={"folder": str(tmp_path)}).json()
        b = chat_client.get("/check_session", params={"folder": str(tmp_path)}).json()

    assert a["session_id"] == b["session_id"]
    assert a["exists"] is False
    assert a["collection_name"] == f"site_{a['session_id']}"


def test_ingest_rejects_missing_folder():
    resp = chat_client.post("/ingest", json={"folder": "/does/not/exist"})
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


def test_chat_init_requires_embeddings():
    """Initializing against a session with no ingested collection is a 409."""
    session = {"folder": None, "collection_name": "site_x", "chat_history": [], "llm": None, "store": None}
    with patch.dict("api.chat_api.sessions", {"s1": session}, clear=True), \
         patch("api.chat_api.collection_exists", return_value=False):
        resp = chat_client.post("/chat/init", json={"session_id": "s1"})

    assert resp.status_code == 409
    assert "ingest" in resp.json()["detail"].lower()


def test_chat_requires_init():
    """A known session that was never initialized returns 409, not 500."""
    session = {"folder": None, "collection_name": "site_x", "chat_history": [], "llm": None, "store": None}
    with patch.dict("api.chat_api.sessions", {"s2": session}, clear=True):
        resp = chat_client.post("/chat", json={"session_id": "s2", "message": "hello"})

    assert resp.status_code == 409
    assert "not initialized" in resp.json()["detail"].lower()


def test_chat_happy_path():
    session = {
        "folder": None,
        "collection_name": "site_x",
        "chat_history": [],
        "llm": MagicMock(),
        "store": MagicMock(),
    }
    with patch.dict("api.chat_api.sessions", {"s3": session}, clear=True), \
         patch("api.chat_api.get_embedder", return_value=MagicMock()), \
         patch("api.chat_api.generate_answer", return_value="the answer") as gen, \
         patch("api.chat_api.save_session"):
        resp = chat_client.post("/chat", json={"session_id": "s3", "message": "hello"})

    assert resp.status_code == 200
    assert resp.json()["response"] == "the answer"
    assert gen.called
    assert session["chat_history"][-1]["assistant"] == "the answer"


def test_chat_surfaces_llm_error_as_502():
    from rag.llm import LLMError

    session = {
        "folder": None,
        "collection_name": "site_x",
        "chat_history": [],
        "llm": MagicMock(),
        "store": MagicMock(),
    }
    with patch.dict("api.chat_api.sessions", {"s4": session}, clear=True), \
         patch("api.chat_api.get_embedder", return_value=MagicMock()), \
         patch("api.chat_api.generate_answer", side_effect=LLMError("bad key")):
        resp = chat_client.post("/chat", json={"session_id": "s4", "message": "hello"})

    assert resp.status_code == 502
    assert "bad key" in resp.json()["detail"]
