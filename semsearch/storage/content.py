import gzip
import hashlib
from pathlib import Path

CONTENT_DIR = Path("data") / "content"
CONTENT_HASH_LEN = 64
LEGACY_CONTENT_HASH_LEN = 16


def content_hash(html: str) -> str:
    return hashlib.sha256(html.encode()).hexdigest()


def content_path(content_hash: str) -> Path:
    return CONTENT_DIR / content_hash[:2] / f"{content_hash[2:]}.html.gz"


def _is_valid_content_hash(content_hash: str) -> bool:
    length = len(content_hash)
    if length not in (LEGACY_CONTENT_HASH_LEN, CONTENT_HASH_LEN):
        return False
    return all(ch in "0123456789abcdef" for ch in content_hash)


def content_available(content_hash: str) -> bool:
    if not _is_valid_content_hash(content_hash):
        return False
    return content_path(content_hash).is_file()


def save_content(html: str) -> str:
    digest = content_hash(html)
    path = content_path(digest)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(html)
    return digest


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
