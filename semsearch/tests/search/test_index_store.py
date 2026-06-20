import pytest
from rank_bm25 import BM25Okapi

from semsearch.search.index_store import dump_index, load_index, load_previous_doc_ids


@pytest.fixture
def index_dir(monkeypatch, tmp_path):
    index_path = tmp_path / "index"
    monkeypatch.setattr("semsearch.search.index_store.INDEX_DIR", index_path)
    monkeypatch.setattr("semsearch.search.index_store.MANIFEST_FILE", index_path / "manifest.json")
    monkeypatch.setattr("semsearch.search.index_store.PAGERANK_FILE", index_path / "pagerank.json")
    monkeypatch.setattr("semsearch.search.index_store.BM25_FILE", index_path / "bm25.pkl")
    return index_path


def test_dump_and_load_index(index_dir):
    bm25 = BM25Okapi([["hello", "world"], ["foo", "bar"]])
    doc_ids = ["doc-a", "doc-b"]
    pagerank = {"doc-a": 1.1, "doc-b": 0.9}

    dump_index(bm25, doc_ids, pagerank)

    loaded_bm25, loaded_doc_ids, loaded_pagerank = load_index()

    assert loaded_doc_ids == doc_ids
    assert loaded_pagerank == pagerank
    assert loaded_bm25.get_scores(["hello"]) == pytest.approx(bm25.get_scores(["hello"]))


def test_load_previous_doc_ids(index_dir):
    assert load_previous_doc_ids() == []

    dump_index(BM25Okapi([["a"]]), ["doc-a"], {"doc-a": 1.0})

    assert load_previous_doc_ids() == ["doc-a"]
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "pagerank.json").exists()
    assert (index_dir / "bm25.pkl").exists()
