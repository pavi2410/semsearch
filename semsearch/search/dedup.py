from ..storage.page import canonical_key


def dedupe_results(
    results: list[tuple[str, float]],
    docs: dict[str, dict[str, str]],
) -> list[tuple[str, float]]:
    """Keep the highest-scoring result per canonical URL."""
    seen: set[str] = set()
    deduped: list[tuple[str, float]] = []

    for doc_id, score in results:
        doc = docs.get(doc_id, {})
        key = canonical_key(doc.get("url", ""), doc.get("canonical_url", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((doc_id, score))

    return deduped
