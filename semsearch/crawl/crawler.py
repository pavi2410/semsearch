import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from rich.live import Live
from rich.progress import Progress, TaskID

from ..core.tui_util import (
    CrawlStats,
    make_crawler_display,
    make_indeterminate_progress,
)
from ..storage import read_content, read_page_meta, save_page
from ..storage.page import normalize_url
from .html_utils import extract_links
from .robots import USER_AGENT, RobotsCache

SEED_URLS = [
    "https://en.wikipedia.org",
    "https://news.ycombinator.com",
]
HTTP_HEADERS = {
    "Accept": "text/html",
    "Accept-Language": "en",
    "User-Agent": USER_AGENT,
}
MAX_CONCURRENT_PER_DOMAIN = 2
RATE_LIMIT_DELAY = 1.0  # minimum seconds between requests to the same domain
REFETCH_INTERVAL = 86400  # seconds before a cached page is considered stale (24h)


def get_rate_limit_wait(
    last_fetch: float | None, delay: float = RATE_LIMIT_DELAY
) -> float:
    """Return how many seconds to wait before the next request to a domain, or 0."""
    if last_fetch is None:
        return 0.0
    return max(0.0, delay - (time.time() - last_fetch))


def is_stale(meta: dict) -> bool:
    """Return True if the page should be re-fetched based on REFETCH_INTERVAL."""
    try:
        parsed = datetime.fromisoformat(meta.get("lastFetchedAt", ""))
        return time.time() - parsed.timestamp() >= REFETCH_INTERVAL
    except ValueError:
        return True


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CrawlerContext:
    progress: Progress
    task_id: TaskID
    client: httpx.AsyncClient
    domain_sems: dict[str, asyncio.Semaphore] = field(default_factory=dict)
    domain_last_fetch: dict[str, float] = field(default_factory=dict)
    robots_cache: RobotsCache = field(init=False)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    visited: set[str] = field(default_factory=set)
    stats: CrawlStats = field(default_factory=CrawlStats)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def __post_init__(self) -> None:
        self.robots_cache = RobotsCache(self.client, default_delay=RATE_LIMIT_DELAY)


async def _fetch_and_save(url: str, ctx: CrawlerContext) -> list[str] | None:
    ctx.stats.inc("in_flight")
    try:
        if ctx.shutdown_event.is_set():
            return None

        if not await ctx.robots_cache.can_fetch(url):
            ctx.progress.print(f"  [yellow]Disallowed by robots.txt[/yellow] {url}")
            ctx.stats.inc("robots_blocked")
            return None

        meta = read_page_meta(url)
        if meta is not None and not is_stale(meta):
            ctx.progress.print(f"  Skip fetching {url}")
            ctx.stats.inc("skipped")
            html = read_content(meta["contentHash"])
            return extract_links(html, url)

        domain = urlparse(url).hostname or url
        if domain not in ctx.domain_sems:
            ctx.domain_sems[domain] = asyncio.Semaphore(MAX_CONCURRENT_PER_DOMAIN)
        sem = ctx.domain_sems[domain]

        async with sem:
            if ctx.shutdown_event.is_set():
                return None

            effective_delay = await ctx.robots_cache.crawl_delay(url)
            rate_limit_wait = get_rate_limit_wait(
                ctx.domain_last_fetch.get(domain), effective_delay
            )
            if rate_limit_wait > 0:
                ctx.progress.print(
                    f"  [dim]Rate limiting {domain} for {rate_limit_wait:.2f}s[/dim]"
                )
                ctx.stats.inc("rate_limited")
                await asyncio.sleep(rate_limit_wait)

            ctx.stats.inc("requests")
            try:
                resp = await ctx.client.get(
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
                ctx.domain_last_fetch[domain] = time.time()

            ctx.stats.inc("req_2xx")
            ctx.stats.inc("req_3xx", by=len(resp.history))
            ctx.stats.inc("saved")
            now = _now()
            save_page(url, html, now)

            final_url = normalize_url(str(resp.url))
            if final_url != url:
                if final_url not in ctx.visited:
                    ctx.visited.add(final_url)
                    ctx.stats.inc("visited")
                save_page(final_url, html, now)
                ctx.progress.print(f"  Saved {url} -> {final_url}")
            else:
                ctx.progress.print(f"  Saved {url}")

            return extract_links(html, url)
    finally:
        ctx.stats.inc("in_flight", by=-1)


async def _worker(ctx: CrawlerContext) -> None:
    while True:
        url = await ctx.queue.get()
        try:
            if ctx.shutdown_event.is_set():
                return
            links = await _fetch_and_save(url, ctx)
            if links:
                ctx.progress.update(ctx.task_id, advance=1)
                for link in links:
                    norm_link = normalize_url(link)
                    if norm_link not in ctx.visited:
                        ctx.visited.add(norm_link)
                        ctx.stats.inc("visited")
                        await ctx.queue.put(norm_link)
        finally:
            ctx.queue.task_done()


async def _crawl(ctx: CrawlerContext, num_workers: int = 10) -> None:
    workers = [asyncio.create_task(_worker(ctx)) for _ in range(num_workers)]
    try:
        await ctx.queue.join()
    except asyncio.CancelledError:
        pass
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


def main() -> None:
    progress = make_indeterminate_progress(
        count_text="{task.completed} pages", unit="pg/s"
    )
    stats = CrawlStats()
    display = make_crawler_display(progress, stats)

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            with Live(display, refresh_per_second=4):
                task_id = progress.add_task("Crawling...", total=None)

                ctx = CrawlerContext(
                    progress=progress,
                    task_id=task_id,
                    client=client,
                    stats=stats,
                )

                for url in SEED_URLS:
                    norm_url = normalize_url(url)
                    ctx.visited.add(norm_url)
                    ctx.stats.inc("visited")
                    await ctx.queue.put(norm_url)

                try:
                    await _crawl(ctx)
                except KeyboardInterrupt:
                    progress.print("[yellow]Shutting down...[/yellow]")
                    ctx.shutdown_event.set()

                progress.update(task_id, description="Crawling complete")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
