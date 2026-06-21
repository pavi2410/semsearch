from semsearch.storage.models import db, init_db
from semsearch.storage.schema_migrate import SemsearchMigrator, run_schema_migrations
from playhouse.migrate import migrate


def test_semsearch_migrator_drops_duplicate_link_index(tmp_path):
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
    migrator = SemsearchMigrator(db)
    with db.transaction():
        migrate(migrator.drop_index("links", "synclink_source_hash_target_url"))

    rows = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'links'"
    ).fetchall()
    assert [row[0] for row in rows] == ["link_source_hash_target_url"]


def test_run_schema_migrations_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    run_schema_migrations()
    run_schema_migrations()

    column_type = db.execute_sql("PRAGMA table_info(token_cache)").fetchall()[1][2]
    assert column_type.upper() == "JSONB"
