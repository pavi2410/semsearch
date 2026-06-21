from peewee import SQL
from playhouse.migrate import SqliteMigrator, migrate, operation

from .models import (
    Block,
    EmbeddingCache,
    Link,
    Page,
    TargetUrl,
    TokenCache,
    _PAGE_COLUMNS,
    db,
)
from .content_migrate import migrate_filesystem_to_content_db
from .vector_codec import is_quantized_embedding


def _table_columns(table: str) -> set[str]:
    rows = db.execute_sql(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _column_type(table: str, column: str) -> str | None:
    rows = db.execute_sql(f"PRAGMA table_info({table})").fetchall()
    for row in rows:
        if row[1] == column:
            return row[2]
    return None


def _link_has_target_foreign_key() -> bool:
    rows = db.execute_sql("PRAGMA foreign_key_list('links')").fetchall()
    return any(row[2] == "target_urls" and row[3] == "target_id" for row in rows)


class SemsearchMigrator(SqliteMigrator):
    @operation
    def ensure_link_target_foreign_key(self):
        link_cols = _table_columns("links")
        if not link_cols or "target_id" not in link_cols:
            return None
        if _link_has_target_foreign_key():
            return None

        return [
            SQL(
                """
                CREATE TABLE "links_fk" (
                    "source_hash" TEXT NOT NULL,
                    "target_id" INTEGER NOT NULL REFERENCES "target_urls" ("id"),
                    UNIQUE ("source_hash", "target_id")
                )
                """
            ),
            SQL(
                """
                INSERT INTO "links_fk" ("source_hash", "target_id")
                SELECT "source_hash", "target_id" FROM "links"
                """
            ),
            SQL('DROP TABLE "links"'),
            SQL('ALTER TABLE "links_fk" RENAME TO "links"'),
            SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS "link_source_hash_target_id"
                ON "links" ("source_hash", "target_id")
                """
            ),
        ]

    @operation
    def drop_legacy_links_table(self):
        link_cols = _table_columns("links")
        if not link_cols or "target_url" not in link_cols:
            return None
        return [SQL('DROP TABLE IF EXISTS "links"')]

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
        if is_quantized_embedding(row[0]):
            return None
        return SQL('DELETE FROM "embedding_cache"')


def run_schema_migrations() -> None:
    """Apply incremental schema changes to an existing semsearch.db."""
    migrator = SemsearchMigrator(db)
    db.create_tables([Page, Block, TokenCache, EmbeddingCache], safe=True)

    page_cols = _table_columns("pages")
    operations = []
    for column in _PAGE_COLUMNS:
        if column not in page_cols:
            operations.append(migrator.add_column("pages", column, Page._meta.fields[column]))

    operations.append(migrator.drop_legacy_links_table())

    if operations:
        with db.transaction():
            migrate(*operations)

    db.create_tables([TargetUrl, Link, TokenCache, EmbeddingCache], safe=True)

    fk_operations = [migrator.ensure_link_target_foreign_key()]
    with db.transaction():
        migrate(*fk_operations)

    operations = [
        migrator.convert_token_cache_tokens_to_jsonb(),
        migrator.clear_legacy_embedding_cache(),
    ]
    with db.transaction():
        migrate(*operations)

    migrate_filesystem_to_content_db()
