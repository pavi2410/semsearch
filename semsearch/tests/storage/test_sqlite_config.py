from semsearch.storage.models import db, init_db
from semsearch.storage.sqlite_config import SQLITE_PRAGMAS, active_pragma


def test_init_db_applies_sqlite_pragmas(tmp_path):
    init_db(tmp_path / "test.db")
    db.connect(reuse_if_open=True)

    assert active_pragma(db, "journal_mode").lower() == "wal"
    assert active_pragma(db, "foreign_keys") == 1
    assert active_pragma(db, "synchronous") == 1  # NORMAL
    assert active_pragma(db, "temp_store") == 2  # MEMORY
    assert active_pragma(db, "cache_size") == SQLITE_PRAGMAS["cache_size"]
    assert active_pragma(db, "mmap_size") == SQLITE_PRAGMAS["mmap_size"]


def test_links_target_id_has_foreign_key(tmp_path):
    init_db(tmp_path / "test.db")

    rows = db.execute_sql("PRAGMA foreign_key_list('links')").fetchall()
    assert rows
    assert rows[0][2] == "target_urls"
    assert rows[0][3] == "target_id"
    assert rows[0][4] == "id"
