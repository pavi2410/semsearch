import time
from collections import deque
from datetime import datetime, timezone

import httpx
from rich.progress import Progress

from .core.html_utils import extract_links
from .core.tui_util import make_indeterminate_progress
from .storage import read_content, read_page_meta, save_page

SEED_URLS = [
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


def _fetch_and_save(
    client: httpx.Client, url: str, progress: Progress, task_id: int
) -> list[str] | None:
    meta = read_page_meta(url)
    if meta is not None:
        last_fetched = meta.get("lastFetchedAt", "")
        try:
            parsed = datetime.fromisoformat(last_fetched)
            if time.time() - parsed.timestamp() < 86400:
                progress.console.print(f"  Skip fetching {url}")
                html = read_content(meta["contentHash"])
                return extract_links(html, url)
        except ValueError:
            pass

    try:
        resp = client.get(url, headers=HTTP_HEADERS, follow_redirects=True, timeout=5)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        progress.console.print(f"  [red]Error[/red] fetching {url}: {e}")
        return None

    save_page(url, html, _now())
    progress.console.print(f"  Saved {url}")
    return extract_links(html, url)


def main() -> None:
    visited: set[str] = set()
    queue: deque[str] = deque(SEED_URLS)

    progress = make_indeterminate_progress(
        count_text="{task.completed} pages", unit="pg/s"
    )

    with progress:
        task_id = progress.add_task("Crawling...", total=None)
        with httpx.Client() as client:
            while queue:
                url = queue.popleft()
                if url in visited:
                    continue

                links = _fetch_and_save(client, url, progress, task_id)
                if links is None:
                    continue

                visited.add(url)
                progress.update(task_id, advance=1, description=f"Crawling {url[:80]}...")

                for link in links:
                    if link not in visited:
                        queue.append(link)

        progress.update(
            task_id,
            description=f"Crawling complete — {len(visited)} pages visited",
        )


if __name__ == "__main__":
    main()
