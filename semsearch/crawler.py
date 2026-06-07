import hashlib
import json
import time
from pathlib import Path

import httpx

from .config import WEBPAGES_DIR
from .html_utils import extract_links
START_URLS = [
    "https://en.wikipedia.org",
    "https://news.ycombinator.com",
]
HTTP_HEADERS = {
    "Accept": "text/html",
    "Accept-Language": "en",
    "User-Agent": "Googlebot",
}


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _file_path(url: str) -> Path:
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname or "unknown"
    return WEBPAGES_DIR / f"{hostname}_{_url_hash(url)}.json"


def _crawl(client: httpx.Client, url: str, visited: set[str]) -> None:
    if url in visited:
        return

    filepath = _file_path(url)

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        last_fetched = data.get("lastFetchedAt", 0)
        if time.time() * 1000 - last_fetched < 86400000:
            print(f"Skip fetching {url}")
            visited.add(url)
            return
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    try:
        resp = client.get(url, headers=HTTP_HEADERS, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return

    WEBPAGES_DIR.mkdir(exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "url": url,
                "lastFetchedAt": int(time.time() * 1000),
                "content": html,
            },
            f,
            ensure_ascii=False,
        )
    print(f"Saved {url} → {filepath.name}")

    visited.add(url)

    for link in extract_links(html, url):
        if link not in visited:
            visited.add(link)
            _crawl(client, link, visited)


def main() -> None:
    print("Crawl starting...")
    visited: set[str] = set()
    with httpx.Client() as client:
        for url in START_URLS:
            _crawl(client, url, visited)
    print(f"Crawl complete. Visited {len(visited)} pages.")


if __name__ == "__main__":
    main()
