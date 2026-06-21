from pathlib import Path

from peewee import (
    AutoField,
    BlobField,
    BooleanField,
    FloatField,
    ForeignKeyField,
    Model,
    SqliteDatabase,
    TextField,
)
from playhouse.sqlite_ext import JSONBField

from .sqlite_config import SQLITE_PRAGMAS, SQLITE_TIMEOUT_SEC

_DB_PATH = Path("data") / "semsearch.db"

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
    db.init(_db_path_str(path), pragmas=SQLITE_PRAGMAS, timeout=SQLITE_TIMEOUT_SEC)
    migrate_schema()
    db.create_tables(
        [Page, Block, TargetUrl, Link, TokenCache, EmbeddingCache],
        safe=True,
    )


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


class TargetUrl(BaseModel):
    id = AutoField(primary_key=True)
    url = TextField(unique=True)

    class Meta:
        table_name = "target_urls"


class Link(BaseModel):
    source_hash = TextField()
    target = ForeignKeyField(TargetUrl, column_name="target_id")

    class Meta:
        table_name = "links"
        primary_key = False
        indexes = ((("source_hash", "target_id"), True),)


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
