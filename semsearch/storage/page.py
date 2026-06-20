import hashlib
from collections.abc import Iterator
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .content import save_content
from .models import SyncPage as Page

_LOCALE_QUERY_KEYS = frozenset({"locale", "lang", "language", "hl"})


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
    if parsed.query:
        filtered = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _LOCALE_QUERY_KEYS
        ]
        parsed = parsed._replace(query=urlencode(filtered, doseq=True))
    return urlunparse(parsed)


def canonical_key(url: str, canonical_url: str = "") -> str:
    preferred = canonical_url.strip() if canonical_url else url
    if not preferred:
        return ""
    return normalize_url(preferred)


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


async def async_save_page(url: str, html: str, last_fetched: str) -> dict:
    from .models import Page as AsyncPage, db

    content_hash = save_content(html)
    url = normalize_url(url)
    h = url_hash(url)
    await db.aexecute(
        AsyncPage.replace(
            url_hash=h,
            url=url,
            fetched_at=last_fetched,
            content_hash=content_hash,
        )
    )
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


async def async_read_page_meta(url: str) -> dict | None:
    from .models import Page as AsyncPage, db

    h = url_hash(normalize_url(url))
    try:
        page = await db.get(AsyncPage.select().where(AsyncPage.url_hash == h))
    except AsyncPage.DoesNotExist:
        return None
    return {
        "url": page.url,
        "lastFetchedAt": page.fetched_at,
        "contentHash": page.content_hash,
    }


def iter_page_metas() -> Iterator[dict]:
    for page in Page.select().iterator():
        yield {
            "url": page.url,
            "urlHash": page.url_hash,
            "lastFetchedAt": page.fetched_at,
            "contentHash": page.content_hash,
            "indexedContentHash": page.indexed_content_hash or "",
        }
