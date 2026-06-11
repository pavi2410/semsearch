from pathlib import Path

from peewee import BooleanField, FloatField, Model, SqliteDatabase, TextField
from playhouse.pwasyncio import AsyncSqliteDatabase

_DB_PATH = Path("data") / "semsearch.db"

_PRAGMAS = {
    "journal_mode": "wal",  # writes don't block reads
}

# Async db used by the crawler (inside asyncio event loop)
db = AsyncSqliteDatabase(None)

# Sync db used by the indexer and searcher (outside event loop)
sync_db = SqliteDatabase(None)


def _db_path_str(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def init_db(path: Path = _DB_PATH) -> None:
    """Initialise the sync database. Used by indexer, searcher, and migrate."""
    sync_db.init(_db_path_str(path), pragmas=_PRAGMAS)
    sync_db.create_tables([SyncPage, SyncBlock], safe=True)


async def async_init_db(path: Path = _DB_PATH) -> None:
    """Initialise the async database. Used by the crawler."""
    db.init(_db_path_str(path), pragmas=_PRAGMAS)
    async with db:
        await db.acreate_tables([Page, Block], safe=True)


class BaseModel(Model):
    class Meta:
        database = db


class Page(BaseModel):
    url_hash = TextField(primary_key=True)
    url = TextField()
    fetched_at = TextField()
    content_hash = TextField()
    title = TextField(null=True)

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


class SyncPage(Model):
    url_hash = TextField(primary_key=True)
    url = TextField()
    fetched_at = TextField()
    content_hash = TextField()
    title = TextField(null=True)

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
