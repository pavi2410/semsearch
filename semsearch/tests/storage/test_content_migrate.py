import gzip
import hashlib
from pathlib import Path

from semsearch.storage.content import content_available, content_hash, try_read_content
from semsearch.storage.content_migrate import (
    LEGACY_CONTENT_HASH_LEN,
    migrate_filesystem_to_content_db,
)
from semsearch.storage.models import EmbeddingCache, Page, TokenCache, init_db


def _sha256_hex(html: str) -> str:
    return hashlib.sha256(html.encode()).hexdigest()


def _legacy_content_path(filesystem_key: str, content_dir: Path) -> Path:
    return content_dir / filesystem_key[:2] / f"{filesystem_key[2:]}.html.gz"


def test_migrate_filesystem_rekeys_pages_to_blake3(tmp_path, monkeypatch):
    content_dir = tmp_path / "content"
    monkeypatch.setattr("semsearch.storage.content_migrate.CONTENT_DIR", content_dir)

    db_path = tmp_path / "test.db"
    init_db(db_path)

    html = "<html><body>Legacy</body></html>"
    sha_key = _sha256_hex(html)
    path = _legacy_content_path(sha_key, content_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(html)

    Page.create(
        url_hash="doc1",
        url="https://example.com/",
        fetched_at="2026-01-01T00:00:00Z",
        content_hash=sha_key,
        indexed_content_hash=sha_key,
    )
    TokenCache.create(content_hash=sha_key, tokens=["legacy"])
    EmbeddingCache.create(content_hash=sha_key, payload=b"legacy")

    imported = migrate_filesystem_to_content_db()
    assert imported == 1

    page = Page.get_by_id("doc1")
    blake3_key = content_hash(html)
    assert page.content_hash == blake3_key
    assert page.indexed_content_hash == blake3_key
    assert try_read_content(blake3_key) == html
    assert TokenCache.select().count() == 0
    assert EmbeddingCache.select().count() == 0
    assert not content_dir.exists()


def test_migrate_filesystem_handles_16_char_paths(tmp_path, monkeypatch):
    content_dir = tmp_path / "content"
    monkeypatch.setattr("semsearch.storage.content_migrate.CONTENT_DIR", content_dir)

    db_path = tmp_path / "test.db"
    init_db(db_path)

    html = "<html><body>Short key</body></html>"
    sha_key = _sha256_hex(html)
    legacy_key = sha_key[:LEGACY_CONTENT_HASH_LEN]
    path = _legacy_content_path(legacy_key, content_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(html)

    Page.create(
        url_hash="doc1",
        url="https://example.com/",
        fetched_at="2026-01-01T00:00:00Z",
        content_hash=legacy_key,
    )

    imported = migrate_filesystem_to_content_db()
    assert imported == 1
    assert Page.get_by_id("doc1").content_hash == content_hash(html)


def test_migrate_filesystem_hash_mismatch_keeps_content_dir(tmp_path, monkeypatch):
    content_dir = tmp_path / "content"
    monkeypatch.setattr("semsearch.storage.content_migrate.CONTENT_DIR", content_dir)

    db_path = tmp_path / "test.db"
    init_db(db_path)

    html = "<html><body>Good</body></html>"
    wrong_key = "0" * 64
    path = _legacy_content_path(wrong_key, content_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(html)

    Page.create(
        url_hash="doc1",
        url="https://example.com/",
        fetched_at="2026-01-01T00:00:00Z",
        content_hash=wrong_key,
    )

    imported = migrate_filesystem_to_content_db()
    assert imported == 0
    assert content_dir.exists()
    assert path.is_file()


def test_migrate_filesystem_skips_orphan_hash_mismatch(tmp_path, monkeypatch):
    content_dir = tmp_path / "content"
    monkeypatch.setattr("semsearch.storage.content_migrate.CONTENT_DIR", content_dir)

    db_path = tmp_path / "test.db"
    init_db(db_path)

    html = "<html><body>Orphan</body></html>"
    wrong_key = "0" * 64
    path = _legacy_content_path(wrong_key, content_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(html)

    assert migrate_filesystem_to_content_db() == 0
    assert not content_dir.exists()


def test_migrate_filesystem_idempotent(tmp_path, monkeypatch):
    content_dir = tmp_path / "content"
    monkeypatch.setattr("semsearch.storage.content_migrate.CONTENT_DIR", content_dir)

    db_path = tmp_path / "test.db"
    init_db(db_path)

    html = "<html><body>Once</body></html>"
    sha_key = _sha256_hex(html)
    path = _legacy_content_path(sha_key, content_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(html)

    Page.create(
        url_hash="doc1",
        url="https://example.com/",
        fetched_at="2026-01-01T00:00:00Z",
        content_hash=sha_key,
    )

    assert migrate_filesystem_to_content_db() == 1
    assert migrate_filesystem_to_content_db() == 0
    assert content_available(content_hash(html))
