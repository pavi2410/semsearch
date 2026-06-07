import hashlib
import json
import time
from pathlib import Path

import httpx

from .core.config import WEBPAGES_DIR
from .core.html_utils import extract_links
from .core.tui_util import make_indeterminate_progress

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


def _crawl(
    client: httpx.Client,
    url: str,
    visited: set[str],
    progress: Progress,
    task_id: int,
) -> None:
    if url in visited:
        return

    filepath = _file_path(url)

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        last_fetched = data.get("lastFetchedAt", 0)
        if time.time() * 1000 - last_fetched < 86400000:
            progress.console.print(f"  Skip fetching {url}")
            visited.add(url)
            return
    except FileNotFoundError, json.JSONDecodeError:
        pass

    try:
        resp = client.get(url, headers=HTTP_HEADERS, follow_redirects=True, timeout=5)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        progress.console.print(f"  [red]Error[/red] fetching {url}: {e}")
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
    progress.console.print(f"  Saved {url} → {filepath.name}")

    visited.add(url)
    progress.update(task_id, advance=1, description=f"Crawling {url[:80]}...")

    for link in extract_links(html, url):
        if link not in visited:
            visited.add(link)
            _crawl(client, link, visited, progress, task_id)


def main() -> None:
    visited: set[str] = set()

    progress = make_indeterminate_progress(
        count_text="{task.completed} pages", unit="pg/s"
    )

    with progress:
        task_id = progress.add_task("Crawling...", total=None)
        with httpx.Client() as client:
            for url in START_URLS:
                if url not in visited:
                    visited.add(url)
                    _crawl(client, url, visited, progress, task_id)
        progress.update(
            task_id,
            description=f"Crawling complete — {len(visited)} pages visited",
        )


if __name__ == "__main__":
    main()
