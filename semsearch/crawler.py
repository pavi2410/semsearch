import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from rich.progress import Progress

from .core.html_utils import extract_links
from .core.thread_utils import ThreadSafeDict, ThreadSafeSet
from .core.tui_util import make_indeterminate_progress
from .storage import read_content, read_page_meta, save_page
from .storage.page import normalize_url

SEED_URLS = [
    "https://en.wikipedia.org",
    "https://news.ycombinator.com",
]
HTTP_HEADERS = {
    "Accept": "text/html",
    "Accept-Language": "en",
    "User-Agent": "Googlebot",
}
MAX_WORKERS = 10
MAX_CONCURRENT_PER_DOMAIN = 2

_thread_local = threading.local()


def _get_client() -> httpx.Client:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = httpx.Client()
    return _thread_local.client


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_and_save(
    url: str,
    progress: Progress,
    task_id: int,
    domain_sems: ThreadSafeDict[str, threading.Semaphore],
    shutdown_event: threading.Event,
    visited: "ThreadSafeSet[str]",
) -> list[str] | None:
    client = _get_client()

    if shutdown_event.is_set():
        return None

    meta = read_page_meta(url)
    if meta is not None:
        last_fetched = meta.get("lastFetchedAt", "")
        try:
            parsed = datetime.fromisoformat(last_fetched)
            if time.time() - parsed.timestamp() < 86400:
                progress.print(f"  Skip fetching {url}")
                html = read_content(meta["contentHash"])
                return extract_links(html, url)
        except ValueError:
            pass

    domain = urlparse(url).hostname or url
    sem = domain_sems.get_or_insert(
        domain, lambda: threading.Semaphore(MAX_CONCURRENT_PER_DOMAIN)
    )

    while not sem.acquire(timeout=0.5):
        if shutdown_event.is_set():
            return None

    try:
        if shutdown_event.is_set():
            return None
        try:
            resp = client.get(
                url, headers=HTTP_HEADERS, follow_redirects=True, timeout=5
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            progress.print(f"  [red]Error[/red] fetching {url}: {e}")
            return None

        now = _now()
        save_page(url, html, now)

        final_url = normalize_url(str(resp.url))
        if final_url != url:
            visited.add_if_absent(final_url)
            save_page(final_url, html, now)
            progress.print(f"  Saved {url} -> {final_url}")
        else:
            progress.print(f"  Saved {url}")

        return extract_links(html, url)
    finally:
        sem.release()


def main() -> None:
    visited: ThreadSafeSet[str] = ThreadSafeSet()
    domain_sems: ThreadSafeDict[str, threading.Semaphore] = ThreadSafeDict()
    shutdown_event = threading.Event()

    progress = make_indeterminate_progress(
        count_text="{task.completed} pages", unit="pg/s"
    )

    with progress:
        task_id = progress.add_task("Crawling...", total=None)

        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        pending: set[Future] = set()

        for url in SEED_URLS:
            norm_url = normalize_url(url)
            visited.add(norm_url)
            pending.add(
                executor.submit(
                    _fetch_and_save,
                    norm_url,
                    progress,
                    task_id,
                    domain_sems,
                    shutdown_event,
                    visited,
                )
            )

        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        links = future.result()
                    except Exception:
                        continue
                    if links is None:
                        continue

                    progress.update(task_id, advance=1)

                    for link in links:
                        norm_link = normalize_url(link)
                        if visited.add_if_absent(norm_link):
                            pending.add(
                                executor.submit(
                                    _fetch_and_save,
                                    norm_link,
                                    progress,
                                    task_id,
                                    domain_sems,
                                    shutdown_event,
                                    visited,
                                )
                            )
        except KeyboardInterrupt:
            progress.print("[yellow]Shutting down...[/yellow]")
            shutdown_event.set()
            for f in pending:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

        executor.shutdown()

        progress.update(
            task_id,
            description=f"Crawling complete — {len(visited)} pages visited",
        )


if __name__ == "__main__":
    main()
