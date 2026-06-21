from collections.abc import Iterable

from .models import TargetUrl, db
from .page import normalize_url


def intern_urls(urls: Iterable[str]) -> dict[str, int]:
    """Insert normalized URLs into target_urls and return url -> id mapping."""
    normalized = list(dict.fromkeys(normalize_url(url) for url in urls))
    if not normalized:
        return {}

    with db.atomic():
        TargetUrl.insert_many([{"url": url} for url in normalized]).on_conflict_ignore().execute()

    return {
        row.url: row.id
        for row in TargetUrl.select(TargetUrl.id, TargetUrl.url).where(
            TargetUrl.url.in_(normalized)
        )
    }
