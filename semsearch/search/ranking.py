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


def apply_ranking(bm25_score: float, doc: dict[str, str]) -> float:
    return bm25_score * recency_boost(doc) * https_boost(doc)
