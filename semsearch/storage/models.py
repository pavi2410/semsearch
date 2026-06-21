from pathlib import Path

from peewee import BlobField, BooleanField, FloatField, Model, SqliteDatabase, TextField
from playhouse.pwasyncio import AsyncSqliteDatabase
from playhouse.sqlite_ext import JSONBField

_DB_PATH = Path("data") / "semsearch.db"

_PRAGMAS = {
    "journal_mode": "wal",  # writes don't block reads
}

_PAGE_COLUMNS = {
    "description": "TEXT",
    "canonical_url": "TEXT",
    "og_title": "TEXT",
    "og_description": "TEXT",
    "published_at": "TEXT",
    "modified_at": "TEXT",
    "body_excerpt": "TEXT",
    "jsonld_types": "TEXT",
    "language": "TEXT",
    "indexed_content_hash": "TEXT",
    "etag": "TEXT",
    "http_last_modified": "TEXT",
}

# Async db used by the crawler (inside asyncio event loop)
db = AsyncSqliteDatabase(None)

# Sync db used by the indexer and searcher (outside event loop)
sync_db = SqliteDatabase(None)


def _db_path_str(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def migrate_schema() -> None:
    """Add new columns/tables to existing databases without dropping data."""
    from .schema_migrate import run_schema_migrations

    run_schema_migrations()


def init_db(path: Path = _DB_PATH) -> None:
    """Initialise the sync database. Used by indexer, searcher, and migrate."""
    sync_db.init(_db_path_str(path), pragmas=_PRAGMAS)
    sync_db.create_tables([SyncPage, SyncBlock, SyncLink, SyncTokenCache, SyncEmbeddingCache], safe=True)
    migrate_schema()


async def async_init_db(path: Path = _DB_PATH) -> None:
    """Initialise the async database. Used by the crawler."""
    db.init(_db_path_str(path), pragmas=_PRAGMAS)
    async with db:
        await db.acreate_tables([Page, Block, Link], safe=True)


class BaseModel(Model):
    class Meta:
        database = db


class Page(BaseModel):
    url_hash = TextField(primary_key=True)
    url = TextField()
    fetched_at = TextField()
    content_hash = TextField()
    title = TextField(null=True)
    description = TextField(null=True)
    canonical_url = TextField(null=True)
    og_title = TextField(null=True)
    og_description = TextField(null=True)
    published_at = TextField(null=True)
    modified_at = TextField(null=True)
    body_excerpt = TextField(null=True)
    jsonld_types = TextField(null=True)
    language = TextField(null=True)
    indexed_content_hash = TextField(null=True)
    etag = TextField(null=True)
    http_last_modified = TextField(null=True)

    class Meta:
        table_name = "pages"


class Block(BaseModel):
    key = TextField(primary_key=True)  # domain or url
    kind = TextField()  # 'domain' or 'url'
    reason = TextField()
    permanent = BooleanField()
    until = FloatField(null=True)  # epoch seconds; None if permanent

    class Meta:
        table_name = "blocks"


class Link(BaseModel):
    source_hash = TextField()
    target_url = TextField()

    class Meta:
        table_name = "links"
        primary_key = False


class SyncPage(Model):
    url_hash = TextField(primary_key=True)
    url = TextField()
    fetched_at = TextField()
    content_hash = TextField()
    title = TextField(null=True)
    description = TextField(null=True)
    canonical_url = TextField(null=True)
    og_title = TextField(null=True)
    og_description = TextField(null=True)
    published_at = TextField(null=True)
    modified_at = TextField(null=True)
    body_excerpt = TextField(null=True)
    jsonld_types = TextField(null=True)
    language = TextField(null=True)
    indexed_content_hash = TextField(null=True)
    etag = TextField(null=True)
    http_last_modified = TextField(null=True)

    class Meta:
        table_name = "pages"
        database = sync_db


class SyncBlock(Model):
    key = TextField(primary_key=True)
    kind = TextField()
    reason = TextField()
    permanent = BooleanField()
    until = FloatField(null=True)

    class Meta:
        table_name = "blocks"
        database = sync_db


class SyncLink(Model):
    source_hash = TextField()
    target_url = TextField()

    class Meta:
        table_name = "links"
        database = sync_db
        primary_key = False
        indexes = ((("source_hash", "target_url"), True),)


class SyncTokenCache(Model):
    content_hash = TextField(primary_key=True)
    tokens = JSONBField()

    class Meta:
        table_name = "token_cache"
        database = sync_db


class SyncEmbeddingCache(Model):
    content_hash = TextField(primary_key=True)
    payload = BlobField()

    class Meta:
        table_name = "embedding_cache"
        database = sync_db
