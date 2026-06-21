import pytest

from semsearch.storage.models import SyncTokenCache, init_db, migrate_schema, sync_db
from semsearch.storage.token_cache import load_tokens, save_tokens


@pytest.fixture
def db(tmp_path):
    init_db(tmp_path / "test.db")
    yield


def test_token_cache_round_trip(db):
    save_tokens("abc123", ["one", "two"])
    assert load_tokens("abc123") == ["one", "two"]
    assert load_tokens("missing") is None


def test_token_cache_stores_jsonb_blob(db):
    save_tokens("abc123", ["one", "two"])
    storage_type, blob_len = sync_db.execute_sql(
        "SELECT typeof(tokens), length(tokens) FROM token_cache WHERE content_hash = ?",
        ("abc123",),
    ).fetchone()
    assert storage_type == "blob"
    assert blob_len > 0


def test_migrate_token_cache_text_to_jsonb(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    sync_db.execute_sql("DROP TABLE token_cache")
    sync_db.execute_sql(
        """
        CREATE TABLE token_cache (
            content_hash TEXT NOT NULL PRIMARY KEY,
            tokens TEXT NOT NULL
        )
        """
    )
    sync_db.execute_sql(
        "INSERT INTO token_cache (content_hash, tokens) VALUES (?, ?)",
        ("hash-a", '["hello","world"]'),
    )

    migrate_schema()

    assert load_tokens("hash-a") == ["hello", "world"]
    column_type = sync_db.execute_sql("PRAGMA table_info(token_cache)").fetchall()[1][2]
    assert column_type.upper() == "JSONB"


def test_fresh_token_cache_uses_jsonb_column(tmp_path):
    init_db(tmp_path / "test.db")
    column_type = sync_db.execute_sql("PRAGMA table_info(token_cache)").fetchall()[1][2]
    assert column_type.upper() == "JSONB"
