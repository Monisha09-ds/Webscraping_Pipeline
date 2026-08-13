from unittest.mock import patch

from vectorstore.chunker import chunk_markdown
from vectorstore.store import build_from_sitecontent


def test_chunking_logic():
    """Markdown splits into chunks that carry their doc id."""
    text = "# Header\n\n" + "word " * 100
    chunks = chunk_markdown(text, "doc1", "source1")

    assert len(chunks) >= 1
    assert chunks[0].metadata["doc_id"] == "doc1"
    assert chunks[0].metadata["chunk_id"].startswith("doc1_")


def test_chunk_ids_are_unique():
    text = "# A\n\n" + ("alpha " * 400) + "\n\n# B\n\n" + ("beta " * 400)
    chunks = chunk_markdown(text, "doc1", "source1")

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


@patch("vectorstore.store.ChromaStore")
@patch("vectorstore.store.LocalTextEmbedder")
def test_ingest_pipeline(mock_embedder, mock_store, mock_sitecontent_dir, tmp_path):
    """Full ingest flow with the DB and the embedder mocked out."""
    mock_embedder.return_value.embed_documents.return_value = [[0.1] * 8]
    mock_store_instance = mock_store.return_value

    persist_dir = tmp_path / "chroma_store"
    persist_dir.mkdir(parents=True, exist_ok=True)

    build_from_sitecontent(
        site_dir=mock_sitecontent_dir,
        persist_dir=persist_dir,
        embed_model_path="fake-model",
        min_chars=0,
        collection="site_test",
    )

    assert mock_store_instance.add.called
    assert mock_embedder.return_value.embed_documents.call_count == 1
    # The collection name must be threaded through to the store.
    assert mock_store.call_args.kwargs["collection"] == "site_test"


@patch("vectorstore.store.ChromaStore")
@patch("vectorstore.store.LocalTextEmbedder")
def test_ingest_writes_per_collection_stats(
    mock_embedder, mock_store, mock_sitecontent_dir, tmp_path
):
    """Stats are namespaced per collection so two sites don't overwrite each other."""
    mock_embedder.return_value.embed_documents.return_value = [[0.1] * 8]

    persist_dir = tmp_path / "chroma_store"
    persist_dir.mkdir(parents=True, exist_ok=True)

    build_from_sitecontent(
        site_dir=mock_sitecontent_dir,
        persist_dir=persist_dir,
        embed_model_path="fake-model",
        min_chars=0,
        collection="site_abc",
    )

    assert (persist_dir / "stats_site_abc.json").exists()


def test_ingest_rejects_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    try:
        build_from_sitecontent(site_dir=empty, persist_dir=tmp_path / "chroma")
    except RuntimeError as e:
        assert "No markdown" in str(e)
    else:
        raise AssertionError("expected RuntimeError for a directory with no markdown")
