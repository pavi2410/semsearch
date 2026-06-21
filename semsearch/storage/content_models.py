from pathlib import Path

from peewee import BlobField, Model, SqliteDatabase, TextField

from .sqlite_config import CONTENT_DB_PRAGMAS, SQLITE_TIMEOUT_SEC

_CONTENT_DB_PATH = Path("data") / "content.db"

content_db = SqliteDatabase(None)


def _db_path_str(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


class ContentBaseModel(Model):
    class Meta:
        database = content_db


class ContentBlob(ContentBaseModel):
    content_hash = TextField(primary_key=True)
    body = BlobField()

    class Meta:
        table_name = "content"


def init_content_db(path: Path = _CONTENT_DB_PATH) -> None:
    """Initialise the content-addressable storage database."""
    content_db.init(
        _db_path_str(path),
        pragmas=CONTENT_DB_PRAGMAS,
        timeout=SQLITE_TIMEOUT_SEC,
    )
    content_db.create_tables([ContentBlob], safe=True)


def ensure_content_db() -> None:
    """Open content.db if init_db has not run yet."""
    if content_db.database is not None:
        return
    from .models import main_db_path

    init_content_db(main_db_path().parent / "content.db")
