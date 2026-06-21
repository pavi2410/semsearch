import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from semsearch.index.embeddings import EmbeddingIndex
from semsearch.search.index_store import dump_index, load_index, load_previous_doc_ids


@pytest.fixture
def index_dir(monkeypatch, tmp_path):
    index_path = tmp_path / "index"
    monkeypatch.setattr("semsearch.search.index_store.INDEX_DIR", index_path)
    monkeypatch.setattr("semsearch.search.index_store.MANIFEST_FILE", index_path / "manifest.json")
    monkeypatch.setattr("semsearch.search.index_store.PAGERANK_FILE", index_path / "pagerank.json")
    monkeypatch.setattr("semsearch.search.index_store.BM25_FILE", index_path / "bm25.pkl")
    monkeypatch.setattr(
        "semsearch.search.index_store.EMBEDDINGS_FILE", index_path / "embeddings.pkl"
    )
    return index_path


def test_dump_and_load_index(index_dir):
    bm25 = BM25Okapi([["hello", "world"], ["foo", "bar"]])
    doc_ids = ["doc-a", "doc-b"]
    pagerank = {"doc-a": 1.1, "doc-b": 0.9}

    dump_index(bm25, doc_ids, pagerank)

    loaded_bm25, loaded_doc_ids, loaded_pagerank, loaded_embeddings = load_index()

    assert loaded_doc_ids == doc_ids
    assert loaded_pagerank == pagerank
    assert loaded_embeddings is None
    assert loaded_bm25.get_scores(["hello"]) == pytest.approx(bm25.get_scores(["hello"]))


def test_dump_and_load_index_with_embeddings(index_dir):
    bm25 = BM25Okapi([["hello", "world"]])
    doc_ids = ["doc-a"]
    pagerank = {"doc-a": 1.0}
    embedding_index = EmbeddingIndex.from_vectors(
        model_name="test-model",
        chunk_doc_ids=["doc-a"],
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    dump_index(bm25, doc_ids, pagerank, embedding_index)

    _, _, _, loaded_embeddings = load_index()

    assert loaded_embeddings is not None
    assert loaded_embeddings.model_name == embedding_index.model_name
    assert loaded_embeddings.chunk_doc_ids == embedding_index.chunk_doc_ids
    assert np.allclose(loaded_embeddings.vectors, embedding_index.vectors)


def test_load_previous_doc_ids(index_dir):
    assert load_previous_doc_ids() == []

    dump_index(BM25Okapi([["a"]]), ["doc-a"], {"doc-a": 1.0})

    assert load_previous_doc_ids() == ["doc-a"]
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "pagerank.json").exists()
    assert (index_dir / "bm25.pkl").exists()


def test_load_index_ignores_embeddings_from_older_manifest(index_dir):
    bm25 = BM25Okapi([["hello", "world"]])
    embedding_index = EmbeddingIndex.from_vectors(
        model_name="test-model",
        chunk_doc_ids=["doc-a"],
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    dump_index(bm25, ["doc-a"], {"doc-a": 1.0}, embedding_index)

    manifest_path = index_dir / "manifest.json"
    manifest = manifest_path.read_text(encoding="utf-8").replace('"version": 3', '"version": 2')
    manifest_path.write_text(manifest, encoding="utf-8")

    _, _, _, loaded_embeddings = load_index()
    assert loaded_embeddings is None
