import numpy as np
import pytest

from semsearch.index.indexer import _load_cached_embeddings
from semsearch.storage.content import save_content
from semsearch.storage.embedding_cache import save_embedding
from semsearch.storage.models import Page, init_db


def test_load_cached_embeddings_skips_html_on_cache_hit(db, monkeypatch):
    html = "<html><head><title>Hi</title></head><body><p>Hello world</p></body></html>"
    digest = save_content(html)
    Page.replace(
        url_hash="doc1",
        url="https://example.com/page",
        fetched_at="2026-01-01T00:00:00Z",
        content_hash=digest,
        indexed_content_hash=digest,
    ).execute()
    save_embedding(digest, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("chunks_for_content should not run on cache hit")

    monkeypatch.setattr(
        "semsearch.index.embeddings.try_read_content",
        fail_if_called,
    )

    doc_embeddings, pending, cached = _load_cached_embeddings(
        ["doc1"],
        force_embeddings=False,
    )

    assert cached == 1
    assert pending == []
    assert len(doc_embeddings["doc1"].chunks) == 2
    assert doc_embeddings["doc1"].vectors.shape == (2, 2)


def test_load_cached_embeddings_reembeds_when_cache_missing(db):
    html = "<html><head><title>Hi</title></head><body><p>Hello world</p></body></html>"
    digest = save_content(html)
    Page.replace(
        url_hash="doc1",
        url="https://example.com/page",
        fetched_at="2026-01-01T00:00:00Z",
        content_hash=digest,
        indexed_content_hash=digest,
    ).execute()

    _, pending, cached = _load_cached_embeddings(
        ["doc1"],
        force_embeddings=False,
    )

    assert cached == 0
    assert pending == [
        {
            "urlHash": "doc1",
            "url": "https://example.com/page",
            "contentHash": digest,
        }
    ]


@pytest.fixture
def db(tmp_path):
    init_db(tmp_path / "test.db")
    yield
