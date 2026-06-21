import gzip
import hashlib
import logging
import shutil
from pathlib import Path

from .content import content_available, content_hash, save_content
from .models import EmbeddingCache, Page, TokenCache, db

logger = logging.getLogger(__name__)

CONTENT_DIR = Path("data") / "content"
LEGACY_CONTENT_HASH_LEN = 16
SHA256_HASH_LEN = 64


def set_storage_paths(main_db_path: Path) -> None:
    """Point filesystem CAS import at `{main_db_path.parent}/content`."""
    global CONTENT_DIR
    CONTENT_DIR = main_db_path.parent / "content"


def _legacy_content_path(filesystem_key: str) -> Path:
    return CONTENT_DIR / filesystem_key[:2] / f"{filesystem_key[2:]}.html.gz"


def _filesystem_key_from_path(path: Path) -> str:
    return path.parent.name + path.name.removesuffix(".html.gz")


def _sha256_hex(html: str) -> str:
    return hashlib.sha256(html.encode()).hexdigest()


def _verify_filesystem_key(html: str, filesystem_key: str) -> bool:
    digest = _sha256_hex(html)
    if len(filesystem_key) == SHA256_HASH_LEN:
        return digest == filesystem_key
    if len(filesystem_key) == LEGACY_CONTENT_HASH_LEN:
        return digest[:LEGACY_CONTENT_HASH_LEN] == filesystem_key
    return False


def _referenced_content_hashes() -> set[str]:
    referenced: set[str] = set()
    for page in Page.select(Page.content_hash, Page.indexed_content_hash):
        referenced.add(page.content_hash)
        indexed = page.indexed_content_hash
        if indexed:
            referenced.add(indexed)
    return referenced


def _migration_needed() -> bool:
    if CONTENT_DIR.is_dir() and any(CONTENT_DIR.rglob("*.html.gz")):
        return True
    for page in Page.select(Page.content_hash, Page.indexed_content_hash):
        if not content_available(page.content_hash):
            return True
        indexed = page.indexed_content_hash
        if indexed and not content_available(indexed):
            return True
    return False


def _verify_pages_in_content_db() -> bool:
    for page in Page.select(Page.content_hash, Page.indexed_content_hash):
        if not content_available(page.content_hash):
            return False
        indexed = page.indexed_content_hash
        if indexed and not content_available(indexed):
            return False
    return True


def migrate_filesystem_to_content_db() -> int:
    """Import gzip files into content.db, re-key pages from SHA-256 to BLAKE3."""
    if not _migration_needed():
        return 0

    if not CONTENT_DIR.is_dir() or not any(CONTENT_DIR.rglob("*.html.gz")):
        return 0

    referenced = _referenced_content_hashes()
    mapping: dict[str, str] = {}
    referenced_failures = 0
    imported = 0

    for path in sorted(CONTENT_DIR.rglob("*.html.gz")):
        filesystem_key = _filesystem_key_from_path(path)
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                html = handle.read()
        except OSError:
            if filesystem_key in referenced:
                logger.warning("Failed to read referenced content %s", path)
                referenced_failures += 1
            else:
                logger.warning("Failed to read orphan content %s", path)
            continue

        if not _verify_filesystem_key(html, filesystem_key):
            if filesystem_key in referenced:
                logger.warning(
                    "Hash mismatch for referenced content %s (key %s)",
                    path,
                    filesystem_key,
                )
                referenced_failures += 1
            else:
                logger.warning(
                    "Skipping orphan with hash mismatch %s (key %s)",
                    path,
                    filesystem_key,
                )
            continue

        mapping[filesystem_key] = content_hash(html)
        imported += 1

    if referenced_failures:
        logger.error(
            "Content migration aborted: %d referenced file(s) failed",
            referenced_failures,
        )
        return 0

    for filesystem_key, blake3_key in mapping.items():
        path = _legacy_content_path(filesystem_key)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            html = handle.read()
        save_content(html)

    with db.atomic():
        for old_hash, new_hash in mapping.items():
            Page.update(content_hash=new_hash).where(
                Page.content_hash == old_hash
            ).execute()
            Page.update(indexed_content_hash=new_hash).where(
                Page.indexed_content_hash == old_hash
            ).execute()

        TokenCache.delete().execute()
        EmbeddingCache.delete().execute()

    if not _verify_pages_in_content_db():
        logger.error("Content migration aborted: page hashes missing from content.db")
        return 0

    shutil.rmtree(CONTENT_DIR)
    logger.info(
        "Content migration complete: imported %d blob(s), removed %s",
        imported,
        CONTENT_DIR,
    )
    return imported
