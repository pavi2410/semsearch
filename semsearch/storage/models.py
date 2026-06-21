from pathlib import Path

from peewee import BlobField, BooleanField, FloatField, Model, SqliteDatabase, TextField
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

db = SqliteDatabase(None)


def _db_path_str(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def migrate_schema() -> None:
    """Add new columns/tables to existing databases without dropping data."""
    from .schema_migrate import run_schema_migrations

    run_schema_migrations()


def init_db(path: Path = _DB_PATH) -> None:
    """Initialise the database."""
    db.init(_db_path_str(path), pragmas=_PRAGMAS)
    db.create_tables([Page, Block, Link, TokenCache, EmbeddingCache], safe=True)
    migrate_schema()


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
        indexes = ((("source_hash", "target_url"), True),)


class TokenCache(BaseModel):
    content_hash = TextField(primary_key=True)
    tokens = JSONBField()

    class Meta:
        table_name = "token_cache"


class EmbeddingCache(BaseModel):
    content_hash = TextField(primary_key=True)
    payload = BlobField()

    class Meta:
        table_name = "embedding_cache"
