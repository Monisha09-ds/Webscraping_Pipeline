import pytest
from unittest.mock import MagicMock, patch
from vectorstore.chunker import chunk_markdown
from vectorstore.store import build_from_sitecontent

def test_chunking_logic():
    """Test text chunking with overlap."""
    text = "# Header\n\n" + "word " * 100
    chunks = chunk_markdown(text, "doc1", "source1")
    
    assert len(chunks) >= 1
    # Check structure
    assert chunks[0].metadata["doc_id"] == "doc1"

@patch("vectorstore.store.ChromaStore")
@patch("vectorstore.store.LocalTextEmbedder")
def test_ingest_pipeline(mock_embedder, mock_store, mock_sitecontent_dir, tmp_path):
    """Test the full ingest flow with mocked DB and Embedder."""
    # Setup mocks
    mock_embedder_instance = mock_embedder.return_value
    mock_embedder_instance.embed_chunks.return_value = [[0.1]*384]
    
    mock_store_instance = mock_store.return_value
    
    # Manually create the directory since the mocked ChromaStore won't
    (tmp_path / "chroma_store").mkdir(parents=True, exist_ok=True)
    
    # Run ingest
    build_from_sitecontent(
        site_dir=mock_sitecontent_dir,
        persist_dir=tmp_path / "chroma_store",
        embed_model_path=MagicMock(),
        min_chars=0
    )
    
    # Verify interactions
    # build_from_sitecontent calls store.add
    assert mock_store_instance.add.called
    # And uses embedder
    assert mock_embedder_instance.embed_documents.call_count >= 1
