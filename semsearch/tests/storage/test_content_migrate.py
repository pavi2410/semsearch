import gzip

from semsearch.storage.content import CONTENT_DIR, content_hash, content_path
from semsearch.storage.content_migrate import migrate_legacy_content_hashes
from semsearch.storage.models import Page, TokenCache, init_db


def test_migrate_legacy_content_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr("semsearch.storage.content.CONTENT_DIR", tmp_path)
    db_path = tmp_path / "test.db"
    init_db(db_path)

    html = "<html><body>Legacy</body></html>"
    legacy = content_hash(html)[:16]
    path = content_path(legacy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(html)

    Page.create(
        url_hash="doc1",
        url="https://example.com/",
        fetched_at="2026-01-01T00:00:00Z",
        content_hash=legacy,
        indexed_content_hash=legacy,
    )
    TokenCache.create(content_hash=legacy, tokens=["legacy"])

    migrated = migrate_legacy_content_hashes()
    assert migrated == 1

    page = Page.get_by_id("doc1")
    full = content_hash(html)
    assert page.content_hash == full
    assert page.indexed_content_hash == full
    assert not content_path(legacy).is_file()
    assert content_path(full).is_file()
    assert TokenCache.select().count() == 0
