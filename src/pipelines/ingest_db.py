# src/pipelines/inspect_db.py
import sys
from pathlib import Path

# ---- Path bootstrap ------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[3]     
SRC_DIR   = PROJ_ROOT / "webscraper_pipeline" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vectorstore.store import ChromaStore
from utils.config import CHROMA_ROOT, COLLECTION_NAME

def main():
    store = ChromaStore(persist_dir=CHROMA_ROOT)
    print("Docs in DB:", store.count())
    print("Persistence dir:", CHROMA_ROOT)
if __name__ == "__main__":
    main()

