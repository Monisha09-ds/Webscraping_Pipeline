import pytest
from unittest.mock import MagicMock

from rag.answer import NO_ANSWER, generate_answer
from rag.llm import LLMError
from rag.retrieve import retrieve_chunks


def test_retrieve_normalizes_results():
    """Chunks come back with text and a resolved title."""
    mock_store = MagicMock()
    mock_store.search.return_value = [
        {"text": "chunk1", "metadata": {"title": "Doc1"}},
        {"text": "chunk2", "metadata": {"title": "Doc2"}},
    ]

    results = retrieve_chunks("query", mock_store, MagicMock())

    assert len(results) == 2
    assert results[0]["text"] == "chunk1"
    assert results[0]["title"] == "Doc1"


def test_retrieve_reads_chroma_meta_key():
    """ChromaStore.query returns `meta`, not `metadata` - both must work."""
    mock_store = MagicMock()
    mock_store.search.return_value = [{"text": "chunk", "meta": {"title": "FromMeta"}}]

    results = retrieve_chunks("query", mock_store, MagicMock())

    assert results[0]["title"] == "FromMeta"


def test_retrieve_drops_empty_text():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        {"text": "   ", "meta": {}},
        {"text": "real", "meta": {}},
    ]

    results = retrieve_chunks("query", mock_store, MagicMock())

    assert len(results) == 1
    assert results[0]["text"] == "real"


def test_retrieve_rejects_empty_query():
    with pytest.raises(ValueError):
        retrieve_chunks("   ", MagicMock(), MagicMock())


def test_generate_answer():
    mock_store = MagicMock()
    mock_store.search.return_value = [{"text": "relevant info", "meta": {"title": "T"}}]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "This is the answer."

    ans = generate_answer("question", mock_store, MagicMock(), 2, mock_llm)

    assert ans == "This is the answer."
    assert mock_llm.generate.called
    # The retrieved chunk and its title must reach the prompt.
    prompt = mock_llm.generate.call_args[0][0]
    assert "relevant info" in prompt
    assert "T" in prompt


def test_generate_answer_without_context_says_unknown():
    """No retrieval hits -> 'I don't know', and the LLM is never called."""
    mock_store = MagicMock()
    mock_store.search.return_value = []
    mock_llm = MagicMock()

    ans = generate_answer("question", mock_store, MagicMock(), 2, mock_llm)

    assert ans == NO_ANSWER
    assert not mock_llm.generate.called


def test_generate_answer_surfaces_llm_failure():
    """An empty model response is an error, not a fake 'I don't know'."""
    mock_store = MagicMock()
    mock_store.search.return_value = [{"text": "info", "meta": {}}]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = ""

    with pytest.raises(LLMError):
        generate_answer("question", mock_store, MagicMock(), 2, mock_llm)
