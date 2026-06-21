import pickle

import numpy as np
import pytest

from semsearch.index.embeddings import chunks_for_content
from semsearch.storage.embedding_cache import load_embedding, save_embedding
from semsearch.storage.models import SyncEmbeddingCache, init_db, migrate_schema, sync_db


@pytest.fixture
def db(tmp_path):
    init_db(tmp_path / "test.db")
    yield


def test_embedding_cache_vectors_round_trip(db):
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    save_embedding("abc123", vectors)
    loaded = load_embedding("abc123")
    assert loaded is not None
    assert np.allclose(loaded, vectors)
    assert load_embedding("missing") is None


def test_load_embedding_rejects_legacy_chunk_payload(db):
    legacy = pickle.dumps({"chunks": ["a"], "vectors": np.asarray([[1.0, 0.0]], dtype=np.float32)})
    SyncEmbeddingCache.replace(content_hash="legacy", payload=legacy).execute()
    assert load_embedding("legacy") is None


def test_migrate_schema_clears_legacy_embedding_cache(tmp_path):
    init_db(tmp_path / "test.db")
    legacy = pickle.dumps(
        {"chunks": ["a"], "vectors": np.asarray([[1.0, 0.0]], dtype=np.float32)}
    )
    SyncEmbeddingCache.replace(content_hash="legacy", payload=legacy).execute()

    migrate_schema()

    assert SyncEmbeddingCache.select().count() == 0


def test_migrate_schema_keeps_vector_only_embedding_cache(tmp_path):
    init_db(tmp_path / "test.db")
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    save_embedding("ok", vectors)

    migrate_schema()

    assert SyncEmbeddingCache.select().count() == 1
    assert np.allclose(load_embedding("ok"), vectors)


def test_chunks_for_content_round_trip(tmp_path, monkeypatch):
    html = "<html><head><title>Hi</title></head><body><p>Hello world</p></body></html>"
    monkeypatch.setattr(
        "semsearch.index.embeddings.try_read_content",
        lambda content_hash: html if content_hash == "abc123" else None,
    )
    chunks = chunks_for_content("abc123", "https://example.com/page")
    assert chunks
    assert any("hello" in chunk.lower() for chunk in chunks)
