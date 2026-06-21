from peewee import SQL
from playhouse.migrate import SqliteMigrator, migrate, operation

import pickle

from .models import (
    EmbeddingCache,
    Link,
    Page,
    TokenCache,
    _PAGE_COLUMNS,
    db,
)


def _table_columns(table: str) -> set[str]:
    rows = db.execute_sql(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _column_type(table: str, column: str) -> str | None:
    rows = db.execute_sql(f"PRAGMA table_info({table})").fetchall()
    for row in rows:
        if row[1] == column:
            return row[2]
    return None


def _link_index_names() -> set[str]:
    rows = db.execute_sql(
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

    @operation
    def clear_legacy_embedding_cache(self):
        row = self.database.execute_sql(
            "SELECT payload FROM embedding_cache LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        payload = pickle.loads(row[0])
        if not isinstance(payload, dict):
            return None
        return SQL('DELETE FROM "embedding_cache"')


def run_schema_migrations() -> None:
    """Apply incremental schema changes to an existing semsearch.db."""
    migrator = SemsearchMigrator(db)
    operations = []

    page_cols = _table_columns("pages")
    for column in _PAGE_COLUMNS:
        if column not in page_cols:
            operations.append(migrator.add_column("pages", column, Page._meta.fields[column]))

    db.create_tables([Link, TokenCache, EmbeddingCache], safe=True)

    link_indexes = _link_index_names()
    if (
        "link_source_hash_target_url" in link_indexes
        and "synclink_source_hash_target_url" in link_indexes
    ):
        operations.append(
            migrator.drop_index("links", "synclink_source_hash_target_url")
        )

    operations.append(migrator.convert_token_cache_tokens_to_jsonb())
    operations.append(migrator.clear_legacy_embedding_cache())

    if operations:
        with db.transaction():
            migrate(*operations)
