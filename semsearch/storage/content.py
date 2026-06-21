import gzip
import hashlib
from pathlib import Path

CONTENT_DIR = Path("data") / "content"
CONTENT_HASH_LEN = 16


def content_path(content_hash: str) -> Path:
    return CONTENT_DIR / content_hash[:2] / f"{content_hash[2:]}.html.gz"


def content_available(content_hash: str) -> bool:
    if len(content_hash) != CONTENT_HASH_LEN:
        return False
    return content_path(content_hash).is_file()


def save_content(html: str) -> str:
    content_hash = hashlib.sha256(html.encode()).hexdigest()[:CONTENT_HASH_LEN]
    path = content_path(content_hash)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(html)
    return content_hash


def read_content(content_hash: str) -> str:
    with gzip.open(content_path(content_hash), "rt", encoding="utf-8") as f:
        return f.read()


def try_read_content(content_hash: str) -> str | None:
    if not content_available(content_hash):
        return None
    try:
        return read_content(content_hash)
    except OSError:
        return None
