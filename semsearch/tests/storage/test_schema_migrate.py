from semsearch.storage.models import db, init_db
from semsearch.storage.schema_migrate import SemsearchMigrator, run_schema_migrations
from playhouse.migrate import migrate


def test_run_schema_migrations_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    assert (tmp_path / "content.db").is_file()
    run_schema_migrations()
    run_schema_migrations()

    column_type = db.execute_sql("PRAGMA table_info(token_cache)").fetchall()[1][2]
    assert column_type.upper() == "JSONB"
