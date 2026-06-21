from peewee import SQL
from playhouse.migrate import SqliteMigrator, migrate, operation

from .models import (
    SyncEmbeddingCache,
    SyncLink,
    SyncPage,
    SyncTokenCache,
    _PAGE_COLUMNS,
    sync_db,
)


def _table_columns(table: str) -> set[str]:
    rows = sync_db.execute_sql(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _column_type(table: str, column: str) -> str | None:
    rows = sync_db.execute_sql(f"PRAGMA table_info({table})").fetchall()
    for row in rows:
        if row[1] == column:
            return row[2]
    return None


def _link_index_names() -> set[str]:
    rows = sync_db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'links'"
    ).fetchall()
    return {row[0] for row in rows}


class SemsearchMigrator(SqliteMigrator):
    @operation
    def convert_token_cache_tokens_to_jsonb(self):
        column_type = _column_type("token_cache", "tokens")
        if column_type is None:
            return None
        if column_type.upper() == "JSONB":
            return None

        return [
            SQL(
                """
                CREATE TABLE "token_cache_jsonb" (
                    "content_hash" TEXT NOT NULL PRIMARY KEY,
                    "tokens" JSONB NOT NULL
                )
                """
            ),
            SQL(
                """
                INSERT INTO "token_cache_jsonb" ("content_hash", "tokens")
                SELECT "content_hash", jsonb("tokens")
                FROM "token_cache"
                """
            ),
            SQL('DROP TABLE "token_cache"'),
            SQL('ALTER TABLE "token_cache_jsonb" RENAME TO "token_cache"'),
        ]


def run_schema_migrations() -> None:
    """Apply incremental schema changes to an existing semsearch.db."""
    migrator = SemsearchMigrator(sync_db)
    operations = []

    page_cols = _table_columns("pages")
    for column in _PAGE_COLUMNS:
        if column not in page_cols:
            operations.append(migrator.add_column("pages", column, SyncPage._meta.fields[column]))

    sync_db.create_tables([SyncLink, SyncTokenCache, SyncEmbeddingCache], safe=True)

    link_indexes = _link_index_names()
    if (
        "link_source_hash_target_url" in link_indexes
        and "synclink_source_hash_target_url" in link_indexes
    ):
        operations.append(
            migrator.drop_index("links", "synclink_source_hash_target_url")
        )

    operations.append(migrator.convert_token_cache_tokens_to_jsonb())

    if operations:
        with sync_db.transaction():
            migrate(*operations)
