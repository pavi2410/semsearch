import hashlib
from collections.abc import Iterator
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .content import save_content
from .models import Page

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


def _page_meta(
    page: Page,
    *,
    content_hash: str | None = None,
) -> dict:
    return {
        "url": page.url,
        "lastFetchedAt": page.fetched_at,
        "contentHash": content_hash or page.content_hash,
        "etag": page.etag or "",
        "httpLastModified": page.http_last_modified or "",
    }


def save_page(
    url: str,
    html: str,
    last_fetched: str,
    *,
    etag: str | None = None,
    http_last_modified: str | None = None,
) -> dict:
    content_hash = save_content(html)
    url = normalize_url(url)
    h = url_hash(url)
    Page.replace(
        url_hash=h,
        url=url,
        fetched_at=last_fetched,
        content_hash=content_hash,
        etag=etag,
        http_last_modified=http_last_modified,
    ).execute()
    return {
        "url": url,
        "lastFetchedAt": last_fetched,
        "contentHash": content_hash,
        "etag": etag or "",
        "httpLastModified": http_last_modified or "",
    }


def touch_page(
    url: str,
    last_fetched: str,
    *,
    etag: str | None = None,
    http_last_modified: str | None = None,
) -> None:
    updates: dict[str, str] = {"fetched_at": last_fetched}
    if etag is not None:
        updates["etag"] = etag
    if http_last_modified is not None:
        updates["http_last_modified"] = http_last_modified
    h = url_hash(normalize_url(url))
    Page.update(**updates).where(Page.url_hash == h).execute()


def read_page_meta(url: str) -> dict | None:
    h = url_hash(normalize_url(url))
    try:
        page = Page.get_by_id(h)
    except Page.DoesNotExist:
        return None
    return _page_meta(page)


def iter_page_metas() -> Iterator[dict]:
    for page in Page.select().iterator():
        yield {
            "url": page.url,
            "urlHash": page.url_hash,
            "lastFetchedAt": page.fetched_at,
            "contentHash": page.content_hash,
            "indexedContentHash": page.indexed_content_hash or "",
        }
