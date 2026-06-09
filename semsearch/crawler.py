import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from rich.progress import Progress

from .core.html_utils import extract_links
from .core.thread_utils import ThreadSafeSet
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
    domain_sems: dict[str, threading.Semaphore],
    domain_sems_lock: threading.Lock,
    shutdown_event: threading.Event,
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
    with domain_sems_lock:
        if domain not in domain_sems:
            domain_sems[domain] = threading.Semaphore(MAX_CONCURRENT_PER_DOMAIN)
        sem = domain_sems[domain]

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

        save_page(url, html, _now())
        progress.print(f"  Saved {url}")
        return extract_links(html, url)
    finally:
        sem.release()


def main() -> None:
    visited: ThreadSafeSet[str] = ThreadSafeSet()
    domain_sems: dict[str, threading.Semaphore] = {}
    domain_sems_lock = threading.Lock()
    shutdown_event = threading.Event()

    progress = make_indeterminate_progress(
        count_text="{task.completed} pages", unit="pg/s"
    )

    with progress:
        task_id = progress.add_task("Crawling...", total=None)

        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        pending: set[Future] = set()

        for url in SEED_URLS:
            visited.add(url)
            pending.add(
                executor.submit(
                    _fetch_and_save,
                    url,
                    progress,
                    task_id,
                    domain_sems,
                    domain_sems_lock,
                    shutdown_event,
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
                        if visited.add_if_absent(link):
                            pending.add(
                                executor.submit(
                                    _fetch_and_save,
                                    link,
                                    progress,
                                    task_id,
                                    domain_sems,
                                    domain_sems_lock,
                                    shutdown_event,
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
