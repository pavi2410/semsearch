import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
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


@dataclass
class CrawlerContext:
    progress: Progress
    task_id: int
    domain_sems: ThreadSafeDict[str, threading.Semaphore]
    domain_last_fetch: ThreadSafeDict[str, float]
    shutdown_event: threading.Event
    visited: ThreadSafeSet[str]
    stats: CrawlStats


_thread_local = threading.local()


def _get_client() -> httpx.Client:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = httpx.Client()
    return _thread_local.client


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_and_save(url: str, ctx: CrawlerContext) -> list[str] | None:
    client = _get_client()
    ctx.stats.inc("in_flight")
    try:
        if ctx.shutdown_event.is_set():
            return None

        meta = read_page_meta(url)
        if meta is not None and not is_stale(meta):
            ctx.progress.print(f"  Skip fetching {url}")
            ctx.stats.inc("skipped")
            html = read_content(meta["contentHash"])
            return extract_links(html, url)

        domain = urlparse(url).hostname or url
        sem = ctx.domain_sems.get_or_insert(
            domain, lambda: threading.Semaphore(MAX_CONCURRENT_PER_DOMAIN)
        )

        while not sem.acquire(timeout=0.5):
            if ctx.shutdown_event.is_set():
                return None

        try:
            if ctx.shutdown_event.is_set():
                return None

            rate_limit_wait = get_rate_limit_wait(ctx.domain_last_fetch.get(domain))
            if rate_limit_wait > 0:
                ctx.progress.print(
                    f"  [dim]Rate limiting {domain} for {rate_limit_wait:.2f}s[/dim]"
                )
                ctx.stats.inc("rate_limited")
                time.sleep(rate_limit_wait)

            ctx.stats.inc("requests")
            try:
                resp = client.get(
                    url, headers=HTTP_HEADERS, follow_redirects=True, timeout=5
                )
                resp.raise_for_status()
                html = resp.text
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                ctx.stats.inc("req_4xx" if 400 <= code < 500 else "req_5xx")
                ctx.progress.print(f"  [red]HTTP {code}[/red] {url}")
                return None
            except Exception as e:
                ctx.stats.inc("error_net")
                ctx.progress.print(f"  [red]Error[/red] fetching {url}: {e}")
                return None
            finally:
                ctx.domain_last_fetch.set(domain, time.time())

            ctx.stats.inc("req_2xx")
            ctx.stats.inc("req_3xx", by=len(resp.history))
            ctx.stats.inc("saved")
            now = _now()
            save_page(url, html, now)

            final_url = normalize_url(str(resp.url))
            if final_url != url:
                if ctx.visited.add_if_absent(final_url):
                    ctx.stats.inc("visited")
                save_page(final_url, html, now)
                ctx.progress.print(f"  Saved {url} -> {final_url}")
            else:
                ctx.progress.print(f"  Saved {url}")

            return extract_links(html, url)
        finally:
            sem.release()
    finally:
        ctx.stats.inc("in_flight", by=-1)


def main() -> None:
    progress = make_indeterminate_progress(
        count_text="{task.completed} pages", unit="pg/s"
    )
    stats = CrawlStats()
    display = make_crawler_display(progress, stats)

    with Live(display, refresh_per_second=4):
        task_id = progress.add_task("Crawling...", total=None)

        ctx = CrawlerContext(
            progress=progress,
            task_id=task_id,
            domain_sems=ThreadSafeDict(),
            domain_last_fetch=ThreadSafeDict(),
            shutdown_event=threading.Event(),
            visited=ThreadSafeSet(),
            stats=stats,
        )

        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        pending: set[Future] = set()

        def submit_url(url: str) -> None:
            pending.add(executor.submit(_fetch_and_save, url, ctx))

        for url in SEED_URLS:
            norm_url = normalize_url(url)
            if ctx.visited.add_if_absent(norm_url):
                ctx.stats.inc("visited")
                submit_url(norm_url)

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
                        if ctx.visited.add_if_absent(norm_link):
                            ctx.stats.inc("visited")
                            submit_url(norm_link)
        except KeyboardInterrupt:
            progress.print("[yellow]Shutting down...[/yellow]")
            ctx.shutdown_event.set()
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
