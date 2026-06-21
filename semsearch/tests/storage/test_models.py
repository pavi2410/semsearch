from semsearch.storage.models import Link, db, init_db, migrate_schema


def _link_indexes() -> list[str]:
    rows = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'links'"
    ).fetchall()
    return sorted(row[0] for row in rows)


def test_migrate_schema_drops_duplicate_links_index(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    db.execute_sql(
        'CREATE UNIQUE INDEX IF NOT EXISTS "link_source_hash_target_url" '
        'ON "links" ("source_hash", "target_url")'
    )
    db.execute_sql(
        'CREATE UNIQUE INDEX IF NOT EXISTS "synclink_source_hash_target_url" '
        'ON "links" ("source_hash", "target_url")'
    )
    assert _link_indexes() == [
        "link_source_hash_target_url",
        "synclink_source_hash_target_url",
    ]

    migrate_schema()

    assert _link_indexes() == ["link_source_hash_target_url"]


def test_fresh_db_keeps_single_links_index(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    assert _link_indexes() == ["link_source_hash_target_url"]
