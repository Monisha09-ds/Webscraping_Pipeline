import sys
import os
from pathlib import Path
import pytest

# Add src to pythonpath
PROJ_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

@pytest.fixture
def mock_sitecontent_dir(tmp_path):
    """Create a temporary sitecontent directory with some dummy md files."""
    d = tmp_path / "sitecontent"
    d.mkdir()
    
    domain_dir = d / "example-com"
    domain_dir.mkdir()
    
    (domain_dir / "home").mkdir()
    p1 = domain_dir / "home" / "page-1.md"
    p1.write_text("# Home Page\n\nWelcome to example.com")
    
    return d

@pytest.fixture
def mock_chroma_dir(tmp_path):
    """Create a temporary chroma directory."""
    d = tmp_path / "chroma_store"
    d.mkdir()
    return d
