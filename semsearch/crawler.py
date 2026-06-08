import time
from datetime import datetime, timezone

import httpx
from rich.progress import Progress

from .core.html_utils import extract_links
from .core.tui_util import make_indeterminate_progress
from .storage import read_page_meta, save_page

START_URLS = [
    "https://en.wikipedia.org",
    "https://news.ycombinator.com",
]
HTTP_HEADERS = {
    "Accept": "text/html",
    "Accept-Language": "en",
    "User-Agent": "Googlebot",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _crawl(
    client: httpx.Client,
    url: str,
    visited: set[str],
    progress: Progress,
    task_id: int,
) -> None:
    if url in visited:
        return

    meta = read_page_meta(url)
    if meta is not None:
        last_fetched = meta.get("lastFetchedAt", "")
        try:
            parsed = datetime.fromisoformat(last_fetched)
            if time.time() - parsed.timestamp() < 86400:
                progress.console.print(f"  Skip fetching {url}")
                visited.add(url)
                return
        except ValueError:
            pass

    try:
        resp = client.get(url, headers=HTTP_HEADERS, follow_redirects=True, timeout=5)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        progress.console.print(f"  [red]Error[/red] fetching {url}: {e}")
        return

    save_page(url, html, _now())
    progress.console.print(f"  Saved {url}")

    visited.add(url)
    progress.update(task_id, advance=1, description=f"Crawling {url[:80]}...")

    for link in extract_links(html, url):
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
                _crawl(client, url, visited, progress, task_id)
        progress.update(
            task_id,
            description=f"Crawling complete — {len(visited)} pages visited",
        )


if __name__ == "__main__":
    main()
