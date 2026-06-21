import blake3

from .content_codec import compress_html, decompress_html
from .content_models import ContentBlob, ensure_content_db

CONTENT_HASH_LEN = 64


def content_hash(html: str) -> str:
    return blake3.blake3(html.encode()).hexdigest()


def _is_valid_content_hash(content_hash: str) -> bool:
    if len(content_hash) != CONTENT_HASH_LEN:
        return False
    return all(ch in "0123456789abcdef" for ch in content_hash)


def content_available(content_hash: str) -> bool:
    if not _is_valid_content_hash(content_hash):
        return False
    ensure_content_db()
    return ContentBlob.get_or_none(content_hash=content_hash) is not None


def save_content(html: str) -> str:
    ensure_content_db()
    digest = content_hash(html)
    compressed = compress_html(html)
    ContentBlob.insert(content_hash=digest, body=compressed).on_conflict_ignore().execute()
    return digest


def read_content(content_hash: str) -> str:
    ensure_content_db()
    row = ContentBlob.get_by_id(content_hash)
    return decompress_html(row.body)


def try_read_content(content_hash: str) -> str | None:
    if not content_available(content_hash):
        return None
    try:
        return read_content(content_hash)
    except (OSError, ValueError, ContentBlob.DoesNotExist):
        return None
