import math
from datetime import datetime, timezone


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def effective_timestamp(doc: dict[str, str]) -> datetime | None:
    for field in ("modified_at", "published_at", "fetched_at"):
        ts = _parse_timestamp(doc.get(field, ""))
        if ts is not None:
            return ts
    return None


def recency_boost(
    doc: dict[str, str],
    *,
    now: datetime | None = None,
    half_life_days: float = 180.0,
    min_boost: float = 0.85,
    max_boost: float = 1.15,
) -> float:
    """Return a multiplier in [min_boost, max_boost], fresher pages score higher."""
    ts = effective_timestamp(doc)
    if ts is None:
        return 1.0

    reference = now or datetime.now(timezone.utc)
    age_days = max(0.0, (reference - ts).total_seconds() / 86_400)
    freshness = math.exp(-age_days / half_life_days)
    return min_boost + (max_boost - min_boost) * freshness


def https_boost(doc: dict[str, str], *, https_multiplier: float = 1.05, http_multiplier: float = 0.95) -> float:
    url = doc.get("url", "")
    if url.startswith("https://"):
        return https_multiplier
    if url.startswith("http://"):
        return http_multiplier
    return 1.0


def apply_ranking(bm25_score: float, doc: dict[str, str], *, pagerank_boost: float = 1.0) -> float:
    return bm25_score * recency_boost(doc) * https_boost(doc) * pagerank_boost


def compute_pagerank_boosts(
    doc_ids: list[str],
    url_to_doc: dict[str, str],
    links: list[tuple[str, str]],
    *,
    iterations: int = 20,
    damping: float = 0.85,
    min_boost: float = 1.0,
    max_boost: float = 1.3,
) -> dict[str, float]:
    """Compute PageRank over indexed pages and map ranks to score multipliers.

    Raw PageRank on a crawl subgraph is heavily skewed — a few hubs and a long
    tail of low-rank pages. We log-scale before normalizing and use boost-only
    weights so obscure pages stay at 1.0 while linked hubs earn a meaningful lift.
    """
    if not doc_ids:
        return {}

    doc_set = set(doc_ids)
    outlinks: dict[str, list[str]] = {doc_id: [] for doc_id in doc_ids}
    for source_hash, target_url in links:
        if source_hash not in doc_set:
            continue
        target_hash = url_to_doc.get(target_url)
        if target_hash in doc_set:
            outlinks[source_hash].append(target_hash)

    count = len(doc_ids)
    ranks = {doc_id: 1.0 / count for doc_id in doc_ids}

    for _ in range(iterations):
        next_ranks = {doc_id: (1.0 - damping) / count for doc_id in doc_ids}
        for source_hash, targets in outlinks.items():
            share = damping * ranks[source_hash]
            if targets:
                per_target = share / len(targets)
                for target_hash in targets:
                    next_ranks[target_hash] += per_target
            else:
                per_doc = share / count
                for doc_id in doc_ids:
                    next_ranks[doc_id] += per_doc
        ranks = next_ranks

    log_ranks = {doc_id: math.log(ranks[doc_id]) for doc_id in doc_ids}
    min_log = min(log_ranks.values())
    max_log = max(log_ranks.values())
    if max_log == min_log:
        return {doc_id: 1.0 for doc_id in doc_ids}

    log_span = max_log - min_log
    return {
        doc_id: min_boost + (max_boost - min_boost) * (log_ranks[doc_id] - min_log) / log_span
        for doc_id in doc_ids
    }
