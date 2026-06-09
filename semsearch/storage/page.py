import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from .content import save_content

PAGES_DIR = Path("data") / "pages"


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


def _file_path(url: str) -> Path:
    return PAGES_DIR / urlparse(url).hostname / f"{url_hash(url)}.json"


def save_page(url: str, html: str, last_fetched: str) -> dict:
    content_hash = save_content(html)
    url = normalize_url(url)
    meta = {"url": url, "lastFetchedAt": last_fetched, "contentHash": content_hash}
    path = _file_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta


def read_page_meta(url: str) -> dict | None:
    path = _file_path(url)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_page_metas() -> Iterator[dict]:
    for path in sorted(PAGES_DIR.rglob("*.json")):
        with open(path, encoding="utf-8") as f:
            yield json.load(f)
