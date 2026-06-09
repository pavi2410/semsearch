import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from rich.live import Live
from rich.progress import Progress

from .core.html_utils import extract_links
from .core.thread_utils import ThreadSafeDict, ThreadSafeSet
from .core.tui_util import CrawlStats, make_crawler_display, make_indeterminate_progress
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
RATE_LIMIT_DELAY = 1.0  # minimum seconds between requests to the same domain
REFETCH_INTERVAL = 86400  # seconds before a cached page is considered stale (24h)


def get_rate_limit_wait(last_fetch: float | None) -> float:
    """Return how many seconds to wait before the next request to a domain, or 0."""
    if last_fetch is None:
        return 0.0
    return max(0.0, RATE_LIMIT_DELAY - (time.time() - last_fetch))


def is_stale(meta: dict) -> bool:
    """Return True if the page should be re-fetched based on REFETCH_INTERVAL."""
    try:
        parsed = datetime.fromisoformat(meta.get("lastFetchedAt", ""))
        return time.time() - parsed.timestamp() >= REFETCH_INTERVAL
    except ValueError:
        return True


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
    domain_last_fetch: ThreadSafeDict[str, float],
    shutdown_event: threading.Event,
    visited: "ThreadSafeSet[str]",
    stats: CrawlStats,
) -> list[str] | None:
    client = _get_client()
    stats.inc("in_flight")
    try:
        if shutdown_event.is_set():
            return None

        meta = read_page_meta(url)
        if meta is not None and not is_stale(meta):
            progress.print(f"  Skip fetching {url}")
            stats.inc("skipped")
            html = read_content(meta["contentHash"])
            return extract_links(html, url)

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

            rate_limit_wait = get_rate_limit_wait(domain_last_fetch.get(domain))
            if rate_limit_wait > 0:
                progress.print(
                    f"  [dim]Rate limiting {domain} for {rate_limit_wait:.2f}s[/dim]"
                )
                stats.inc("rate_limited")
                time.sleep(rate_limit_wait)

            stats.inc("requests")
            try:
                resp = client.get(
                    url, headers=HTTP_HEADERS, follow_redirects=True, timeout=5
                )
                resp.raise_for_status()
                html = resp.text
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                stats.inc("req_4xx" if 400 <= code < 500 else "req_5xx")
                progress.print(f"  [red]HTTP {code}[/red] {url}")
                return None
            except Exception as e:
                stats.inc("error_net")
                progress.print(f"  [red]Error[/red] fetching {url}: {e}")
                return None
            finally:
                domain_last_fetch.set(domain, time.time())

            stats.inc("req_2xx")
            stats.inc("req_3xx", by=len(resp.history))
            stats.inc("saved")
            now = _now()
            save_page(url, html, now)

            final_url = normalize_url(str(resp.url))
            if final_url != url:
                if visited.add_if_absent(final_url):
                    stats.inc("visited")
                save_page(final_url, html, now)
                progress.print(f"  Saved {url} -> {final_url}")
            else:
                progress.print(f"  Saved {url}")

            return extract_links(html, url)
        finally:
            sem.release()
    finally:
        stats.inc("in_flight", by=-1)


def main() -> None:
    visited: ThreadSafeSet[str] = ThreadSafeSet()
    domain_sems: ThreadSafeDict[str, threading.Semaphore] = ThreadSafeDict()
    domain_last_fetch: ThreadSafeDict[str, float] = ThreadSafeDict()
    shutdown_event = threading.Event()
    stats = CrawlStats()

    progress = make_indeterminate_progress(
        count_text="{task.completed} pages", unit="pg/s"
    )
    display = make_crawler_display(progress, stats)

    with Live(display, refresh_per_second=4):
        task_id = progress.add_task("Crawling...", total=None)

        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        pending: set[Future] = set()

        for url in SEED_URLS:
            norm_url = normalize_url(url)
            visited.add(norm_url)
            stats.inc("visited")
            pending.add(
                executor.submit(
                    _fetch_and_save,
                    norm_url,
                    progress,
                    task_id,
                    domain_sems,
                    domain_last_fetch,
                    shutdown_event,
                    visited,
                    stats,
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
                            stats.inc("visited")
                            pending.add(
                                executor.submit(
                                    _fetch_and_save,
                                    norm_link,
                                    progress,
                                    task_id,
                                    domain_sems,
                                    domain_last_fetch,
                                    shutdown_event,
                                    visited,
                                    stats,
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
            description="Crawling complete",
        )


if __name__ == "__main__":
    main()
