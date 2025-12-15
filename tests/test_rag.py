import pytest
from unittest.mock import MagicMock
from rag.retrieve import retrieve_chunks
from rag.answer import generate_answer

def test_retrieve_context():
    """Test context retrieval from store."""
    mock_store = MagicMock()
    # Mock search to return list of dicts or objects
    mock_store.search.return_value = [
        {"text": "chunk1", "metadata": {"title": "Doc1"}},
        {"text": "chunk2", "metadata": {"title": "Doc2"}}
    ]
    mock_embedder = MagicMock()
    
    results = retrieve_chunks("query", mock_store, mock_embedder)
    
    assert len(results) == 2
    assert results[0]["text"] == "chunk1"

def test_generate_answer():
    """Test answer generation with mocked LLM."""
    mock_store = MagicMock()
    mock_store.search.return_value = [{"text": "relevant info", "metadata": {}}]
    mock_embedder = MagicMock()
    
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "This is the answer."
    
    ans = generate_answer("question", mock_store, mock_embedder, 2, mock_llm)
    
    assert "This is the answer." in ans
    assert mock_llm.generate.called
