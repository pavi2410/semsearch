from __future__ import annotations

from typing import TYPE_CHECKING

from ..storage.page import canonical_key

if TYPE_CHECKING:
    from .search import SearchHit


def dedupe_results(
    results: list[SearchHit],
    docs: dict[str, dict[str, str]],
) -> list[SearchHit]:
    """Keep the highest-scoring result per canonical URL."""
    seen: set[str] = set()
    deduped: list[SearchHit] = []

    for hit in results:
        doc = docs.get(hit.doc_id, {})
        key = canonical_key(doc.get("url", ""), doc.get("canonical_url", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(hit)

    return deduped
