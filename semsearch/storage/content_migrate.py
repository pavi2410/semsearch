from .content import (
    LEGACY_CONTENT_HASH_LEN,
    content_hash,
    content_path,
    read_content,
    save_content,
)
from .models import EmbeddingCache, Page, TokenCache, db


def migrate_legacy_content_hashes() -> int:
    """Upgrade truncated 16-char content hashes to full SHA-256."""
    legacy_hashes: set[str] = set()
    for page in Page.select(Page.content_hash, Page.indexed_content_hash):
        if len(page.content_hash) == LEGACY_CONTENT_HASH_LEN:
            legacy_hashes.add(page.content_hash)
        indexed = page.indexed_content_hash
        if indexed and len(indexed) == LEGACY_CONTENT_HASH_LEN:
            legacy_hashes.add(indexed)

    if not legacy_hashes:
        return 0

    mapping: dict[str, str] = {}
    for old_hash in sorted(legacy_hashes):
        if not content_path(old_hash).is_file():
            continue
        html = read_content(old_hash)
        new_hash = content_hash(html)
        if new_hash != old_hash:
            save_content(html)
        mapping[old_hash] = new_hash

    if not mapping:
        return 0

    with db.transaction():
        for old_hash, new_hash in mapping.items():
            Page.update(content_hash=new_hash).where(
                Page.content_hash == old_hash
            ).execute()
            Page.update(indexed_content_hash=new_hash).where(
                Page.indexed_content_hash == old_hash
            ).execute()
            TokenCache.delete().where(TokenCache.content_hash == old_hash).execute()
            EmbeddingCache.delete().where(
                EmbeddingCache.content_hash == old_hash
            ).execute()
            if old_hash != new_hash:
                old_path = content_path(old_hash)
                if old_path.is_file():
                    old_path.unlink()

    return len(mapping)
