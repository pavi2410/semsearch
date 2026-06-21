import math
from datetime import datetime, timezone

from ..index.nlp import preprocess


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


def metadata_multiplier(doc: dict[str, str], *, pagerank_boost: float = 1.0) -> float:
    return recency_boost(doc) * https_boost(doc) * pagerank_boost


def dampen_metadata_boost(
    metadata_multiplier: float,
    relevance: float,
    *,
    strength: float = 0.85,
) -> float:
    """Reduce hub/recency lift as BM25 relevance rises within a result set."""
    suppression = strength * min(1.0, max(0.0, relevance))
    return 1.0 + (metadata_multiplier - 1.0) * (1.0 - suppression)


def apply_ranking(
    bm25_score: float,
    doc: dict[str, str],
    *,
    pagerank_boost: float = 1.0,
    bm25_max: float | None = None,
) -> float:
    meta = metadata_multiplier(doc, pagerank_boost=pagerank_boost)
    if bm25_max is not None and bm25_max > 0:
        relevance = min(1.0, bm25_score / bm25_max)
        meta = dampen_metadata_boost(meta, relevance)
    return bm25_score * meta


def _has_adjacent_token_sequence(needle: list[str], haystack: list[str]) -> bool:
    if len(needle) < 2 or len(needle) > len(haystack):
        return False
    width = len(needle)
    for index in range(len(haystack) - width + 1):
        if haystack[index : index + width] == needle:
            return True
    return False


def lexical_match_boost(query: str, query_tokens: list[str], doc: dict[str, str]) -> float:
    """Reward title matches, especially full phrases and adjacent query terms."""
    title = doc.get("title", "")
    if not title or not query_tokens:
        return 1.0

    query_lower = query.lower().strip()
    title_lower = title.lower()
    if query_lower and query_lower in title_lower:
        return 1.25

    title_tokens = preprocess(title)
    if _has_adjacent_token_sequence(query_tokens, title_tokens):
        return 1.15

    title_token_set = set(title_tokens)
    if all(token in title_token_set for token in query_tokens):
        return 1.10

    return 1.0


PAGERANK_ITERATIONS = 20


def compute_pagerank_boosts(
    doc_ids: list[str],
    url_to_doc: dict[str, str],
    links: list[tuple[str, str]],
    *,
    iterations: int = PAGERANK_ITERATIONS,
    damping: float = 0.85,
    min_boost: float = 1.0,
    max_boost: float = 1.3,
    progress=None,
    task=None,
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
            if progress is not None and task is not None:
                progress.advance(task)
            continue
        target_hash = url_to_doc.get(target_url)
        if target_hash in doc_set:
            outlinks[source_hash].append(target_hash)
        if progress is not None and task is not None:
            progress.advance(task)

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
        if progress is not None and task is not None:
            progress.advance(task)

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
