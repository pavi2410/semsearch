from semsearch.storage.models import db, init_db, migrate_schema


def _link_indexes() -> list[str]:
    rows = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'links'"
    ).fetchall()
    return sorted(row[0] for row in rows)


def test_fresh_db_uses_interned_links_index(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    assert _link_indexes() == ["link_source_hash_target_id", "link_target_id"]


def test_migrate_schema_drops_legacy_links_table(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    db.execute_sql("DROP TABLE IF EXISTS links")
    db.execute_sql(
        """
        CREATE TABLE links (
            source_hash TEXT NOT NULL,
            target_url TEXT NOT NULL
        )
        """
    )
    db.execute_sql(
        'CREATE UNIQUE INDEX "link_source_hash_target_url" '
        'ON "links" ("source_hash", "target_url")'
    )
    db.execute_sql(
        "INSERT INTO links (source_hash, target_url) "
        "VALUES ('src', 'https://example.com/a')"
    )

    migrate_schema()

    columns = {row[1] for row in db.execute_sql("PRAGMA table_info(links)").fetchall()}
    assert columns == {"source_hash", "target_id"}
    assert db.execute_sql("SELECT COUNT(*) FROM links").fetchone()[0] == 0
    assert _link_indexes() == ["link_source_hash_target_id", "link_target_id"]
