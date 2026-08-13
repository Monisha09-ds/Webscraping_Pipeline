import os
import sys
from pathlib import Path

import pytest

# --- Path + env bootstrap (must happen before any src import) ---
PROJ_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Keep tests off the real Groq API and away from the developer's own .env.
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("LLM_PROVIDER", "groq")


@pytest.fixture
def mock_sitecontent_dir(tmp_path):
    """A sitecontent/ tree containing one scraped page."""
    d = tmp_path / "sitecontent"
    domain_dir = d / "example-com" / "home"
    domain_dir.mkdir(parents=True)

    (domain_dir / "page-1.md").write_text(
        "# Home Page\n\nWelcome to example.com. " + ("content " * 40),
        encoding="utf-8",
    )
    (domain_dir / "_meta.json").write_text(
        '{"url": "http://example.com", "final_url": "http://example.com", '
        '"title": "Example Home"}',
        encoding="utf-8",
    )
    return d


@pytest.fixture
def mock_chroma_dir(tmp_path):
    d = tmp_path / "chroma_store"
    d.mkdir()
    return d


@pytest.fixture
def fake_embedder():
    """Deterministic stand-in for LocalTextEmbedder (no model download)."""
    class FakeEmbedder:
        dimension = 8

        def embed_documents(self, texts, batch_size=None, normalize=True):
            return [[float(len(t) % 7)] * self.dimension for t in texts]

        def embed_query(self, query, normalize=True):
            return [float(len(query) % 7)] * self.dimension

        def cleanup(self):
            pass

    return FakeEmbedder()
