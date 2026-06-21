"""SQLite pragmas applied on every Peewee connection."""

# Negative cache_size values are KiB. ~64 MiB page cache.
_CACHE_SIZE_KIB = 64 * 1024

# Memory-map up to 256 MiB of the database file for read-heavy search.
_MMAP_SIZE_BYTES = 256 * 1024 * 1024

# Wait up to 30s when the database is locked (crawler + indexer overlap).
SQLITE_TIMEOUT_SEC = 30

SQLITE_PRAGMAS = {
    # Concurrent reads during crawl/index writes.
    "journal_mode": "wal",
    # Enforce declared foreign keys (links.target_id -> target_urls.id).
    "foreign_keys": 1,
    # Safe with WAL; faster than FULL fsync on every commit.
    "synchronous": "NORMAL",
    # Keep hot pages in memory for repeated search/index passes.
    "cache_size": -_CACHE_SIZE_KIB,
    # Sort/spill temp tables in RAM during large index builds.
    "temp_store": "MEMORY",
    # Speed up read-only search queries over a large local DB.
    "mmap_size": _MMAP_SIZE_BYTES,
}


def active_pragma(db, name: str):
    """Return the current connection value for a pragma."""
    row = db.execute_sql(f"PRAGMA {name}").fetchone()
    return row[0] if row else None
