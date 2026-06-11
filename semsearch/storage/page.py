import hashlib
from collections.abc import Iterator
from urllib.parse import urlparse, urlunparse

from .content import save_content
from .models import SyncPage as Page


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")
    parsed = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
    )
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    parsed = parsed._replace(path=path)
    return urlunparse(parsed)


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:16]


def save_page(url: str, html: str, last_fetched: str) -> dict:
    content_hash = save_content(html)
    url = normalize_url(url)
    h = url_hash(url)
    Page.replace(
        url_hash=h,
        url=url,
        fetched_at=last_fetched,
        content_hash=content_hash,
    ).execute()
    return {"url": url, "lastFetchedAt": last_fetched, "contentHash": content_hash}


def read_page_meta(url: str) -> dict | None:
    h = url_hash(normalize_url(url))
    try:
        page = Page.get_by_id(h)
        return {
            "url": page.url,
            "lastFetchedAt": page.fetched_at,
            "contentHash": page.content_hash,
        }
    except Page.DoesNotExist:
        return None


def iter_page_metas() -> Iterator[dict]:
    for page in Page.select().iterator():
        yield {
            "url": page.url,
            "lastFetchedAt": page.fetched_at,
            "contentHash": page.content_hash,
        }
