import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from rich import print as rprint
from rich.live import Live
from rich.progress import Progress, TaskID

from ..core.tui_util import (
    CrawlStats,
    make_crawler_display,
    make_indeterminate_progress,
)
from ..storage import (
    async_init_db,
    async_read_page_meta,
    async_save_page,
    async_touch_page,
    init_db,
    read_content,
)
from ..storage.models import db
from ..storage.page import normalize_url
from .blocks import BlockList
from .content_filter import is_fetchable_document_url, is_indexable_page
from .language import detect_page_language, is_crawlable_language
from .metadata import extract_outbound_links
from .robots import USER_AGENT, RobotsCache
from .sitemap import SitemapLoader

SEED_URLS = [
    "https://developer.mozilla.org/en-US/",
    "https://docs.python.org/3/",
    "https://news.ycombinator.com",
    "https://github.com/trending",
    "https://github.com/explore",
]
HTTP_HEADERS = {
    "Accept": "text/html, application/xhtml+xml, text/plain, text/markdown",
    "Accept-Language": "en",
    "User-Agent": USER_AGENT,
}
MAX_CONCURRENT_PER_DOMAIN = 2
RATE_LIMIT_DELAY = 1.0  # minimum seconds between requests to the same domain
REFETCH_INTERVAL = 86400  # seconds before a cached page is considered stale (24h)
SITEMAP_MAX_URLS = 5000  # cap per domain to avoid queue explosion
HTTP_POOL_LIMITS = httpx.Limits(max_connections=30, max_keepalive_connections=15)
AUX_POOL_LIMITS = httpx.Limits(max_connections=15, max_keepalive_connections=8)
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=60.0)
FETCH_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=60.0)
HTTP_GET_RETRIES = 3


def make_page_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(limits=HTTP_POOL_LIMITS, timeout=HTTP_TIMEOUT)


def make_aux_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(limits=AUX_POOL_LIMITS, timeout=HTTP_TIMEOUT)


def cap_sitemap_urls(urls: list[str], limit: int = SITEMAP_MAX_URLS) -> list[str]:
    if len(urls) <= limit:
        return urls
    return urls[:limit]


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


def build_conditional_headers(meta: dict | None) -> dict[str, str]:
    """Build If-None-Match / If-Modified-Since headers from stored page metadata."""
    if meta is None:
        return {}
    headers: dict[str, str] = {}
    etag = meta.get("etag", "").strip()
    if etag:
        headers["If-None-Match"] = etag
    last_modified = meta.get("httpLastModified", "").strip()
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers


def parse_cache_headers(response: httpx.Response) -> tuple[str | None, str | None]:
    etag = response.headers.get("ETag")
    last_modified = response.headers.get("Last-Modified")
    return (
        etag.strip() if etag else None,
        last_modified.strip() if last_modified else None,
    )


def _links_from_cached(meta: dict, url: str) -> list[str]:
    html = read_content(meta["contentHash"])
    return [
        link
        for link in extract_outbound_links(html, url)
        if is_fetchable_document_url(link)
    ]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CrawlerContext:
    progress: Progress
    task_id: TaskID
    client: httpx.AsyncClient
    aux_client: httpx.AsyncClient
    domain_sems: dict[str, asyncio.Semaphore] = field(default_factory=dict)
    domain_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    domain_last_fetch: dict[str, float] = field(default_factory=dict)
    robots_cache: RobotsCache = field(init=False)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    visited: set[str] = field(default_factory=set)
    seen_domains: set[str] = field(default_factory=set)
    stats: CrawlStats = field(default_factory=CrawlStats)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    sitemap_loader: SitemapLoader = field(init=False)
    blocklist: BlockList = field(init=False)
    background_tasks: set[asyncio.Task] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.robots_cache = RobotsCache(self.aux_client, default_delay=RATE_LIMIT_DELAY)
        self.sitemap_loader = SitemapLoader(self.aux_client)
        self.blocklist = BlockList()


def _domain_lock(ctx: CrawlerContext, domain: str) -> asyncio.Lock:
    if domain not in ctx.domain_locks:
        ctx.domain_locks[domain] = asyncio.Lock()
    return ctx.domain_locks[domain]


async def _http_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    timeout: httpx.Timeout,
    ctx: CrawlerContext,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(HTTP_GET_RETRIES):
        try:
            return await client.get(
                url,
                headers=headers,
                follow_redirects=True,
                timeout=timeout,
            )
        except httpx.PoolTimeout as exc:
            last_error = exc
            if attempt + 1 >= HTTP_GET_RETRIES:
                break
            wait = float(attempt + 1)
            ctx.progress.print(
                f"  [dim]Pool busy, retrying in {wait:.0f}s[/dim] {url}"
            )
            ctx.stats.inc("pool_retries")
            await asyncio.sleep(wait)
    assert last_error is not None
    raise last_error


async def _load_domain_sitemap(
    ctx: CrawlerContext, domain: str, sitemap_urls: list[str]
) -> None:
    try:
        page_urls = await ctx.sitemap_loader.load(domain, sitemap_urls)
        capped = cap_sitemap_urls(page_urls)
        if len(page_urls) > len(capped):
            ctx.progress.print(
                f"  [dim]Sitemap {domain}: capped at {len(capped):,} URLs[/dim]"
            )
        ctx.progress.print(f"  [cyan]Sitemap[/cyan] {domain}: {len(capped):,} URLs")
        ctx.stats.inc("sitemap_urls", by=len(capped))
        for page_url in capped:
            if ctx.shutdown_event.is_set():
                return
            norm = normalize_url(page_url)
            if norm not in ctx.visited and is_fetchable_document_url(norm):
                ctx.visited.add(norm)
                ctx.stats.inc("visited")
                await ctx.queue.put(norm)
    except Exception as e:
        ctx.progress.print(f"  [red]Sitemap error[/red] {domain}: {e}")


def _track_background_task(ctx: CrawlerContext, coro) -> None:
    task = asyncio.create_task(coro)
    ctx.background_tasks.add(task)
    task.add_done_callback(ctx.background_tasks.discard)


async def _enqueue_url(url: str, ctx: CrawlerContext) -> None:
    """Add a URL to the queue if not visited. On new domains, bulk-enqueue sitemap URLs."""
    if url in ctx.visited:
        return
    if not is_fetchable_document_url(url):
        return
    ctx.visited.add(url)
    ctx.stats.inc("visited")
    await ctx.queue.put(url)

    domain = urlparse(url).hostname
    if domain:
        async with _domain_lock(ctx, domain):
            if domain in ctx.seen_domains:
                return
            ctx.seen_domains.add(domain)
            ctx.stats.inc("domains_discovered")
            sitemap_urls = await ctx.robots_cache.sitemaps(domain)
        _track_background_task(
            ctx, _load_domain_sitemap(ctx, domain, sitemap_urls)
        )


def _parse_retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except Exception:
        return None


async def _fetch_and_save(url: str, ctx: CrawlerContext) -> list[str] | None:
    ctx.stats.inc("in_flight")
    try:
        if ctx.shutdown_event.is_set():
            return None

        if not await ctx.robots_cache.can_fetch(url):
            ctx.progress.print(f"  [yellow]Disallowed by robots.txt[/yellow] {url}")
            ctx.stats.inc("robots_blocked")
            return None

        blocked, reason = await ctx.blocklist.is_blocked(url)
        if blocked:
            ctx.progress.print(f"  [red]Blocked[/red] ({reason}) {url}")
            ctx.stats.inc("blocked")
            return None

        meta = await async_read_page_meta(url)
        if meta is not None and not is_stale(meta):
            ctx.progress.print(f"  Skip fetching {url}")
            ctx.stats.inc("skipped")
            return _links_from_cached(meta, url)

        is_new_page = meta is None
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
            request_headers = {**HTTP_HEADERS, **build_conditional_headers(meta)}
            try:
                resp = await _http_get_with_retry(
                    ctx.client,
                    url,
                    headers=request_headers,
                    timeout=FETCH_TIMEOUT,
                    ctx=ctx,
                )
            except Exception as e:
                ctx.stats.inc("error_net")
                ctx.progress.print(f"  [red]Error[/red] fetching {url}: {e}")
                return None
            finally:
                ctx.domain_last_fetch[domain] = time.time()

            if resp.status_code == 304:
                if meta is None:
                    ctx.progress.print(f"  [yellow]Unexpected 304[/yellow] {url}")
                    return None
                ctx.stats.inc("not_modified")
                ctx.stats.inc("req_3xx")
                now = _now()
                resp_etag, resp_last_modified = parse_cache_headers(resp)
                await async_touch_page(
                    url,
                    now,
                    etag=resp_etag or meta.get("etag") or None,
                    http_last_modified=resp_last_modified
                    or meta.get("httpLastModified")
                    or None,
                )
                ctx.progress.print(f"  Not modified {url}")
                return _links_from_cached(meta, url)

            try:
                resp.raise_for_status()
                body = resp.text
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                ctx.stats.inc("req_4xx" if 400 <= code < 500 else "req_5xx")
                ctx.progress.print(f"  [red]HTTP {code}[/red] {url}")
                retry_after = _parse_retry_after(e.response)
                await ctx.blocklist.record(url, code, retry_after)
                return None

            ctx.stats.inc("req_2xx")
            ctx.stats.inc("req_3xx", by=len(resp.history))
            content_type = resp.headers.get("Content-Type")
            etag, http_last_modified = parse_cache_headers(resp)

        if not is_indexable_page(url, body, content_type):
            ctx.progress.print(
                f"  [yellow]Skipping unsupported content[/yellow] {url}"
                f" ({content_type or 'unknown type'})"
            )
            ctx.stats.inc("skipped")
            return None

        language = detect_page_language(body, content_type)
        if not is_crawlable_language(language):
            ctx.progress.print(
                f"  [yellow]Skipping non-English[/yellow] {url} ({language})"
            )
            ctx.stats.inc("language_skipped")
            return None

        ctx.stats.inc("pages_new" if is_new_page else "pages_refreshed")
        now = _now()
        await async_save_page(
            url,
            body,
            now,
            etag=etag,
            http_last_modified=http_last_modified,
        )

        final_url = normalize_url(str(resp.url))
        if final_url != url:
            if final_url not in ctx.visited:
                ctx.visited.add(final_url)
                ctx.stats.inc("visited")
            await async_save_page(
                final_url,
                body,
                now,
                etag=etag,
                http_last_modified=http_last_modified,
            )
            ctx.progress.print(f"  Saved {url} -> {final_url}")
        else:
            ctx.progress.print(f"  Saved {url}")

        return [
            link
            for link in extract_outbound_links(body, url)
            if is_fetchable_document_url(link)
        ]
    finally:
        ctx.stats.inc("in_flight", by=-1)


async def _worker(ctx: CrawlerContext) -> None:
    while True:
        url = await ctx.queue.get()
        try:
            if ctx.shutdown_event.is_set():
                return
            try:
                links = await _fetch_and_save(url, ctx)
            except Exception as e:
                ctx.progress.print(f"  [red]Error[/red] processing {url}: {e}")
                links = None
            if links is not None:
                ctx.progress.update(ctx.task_id, advance=1)
                for link in links:
                    await _enqueue_url(normalize_url(link), ctx)
        finally:
            ctx.queue.task_done()


async def _crawl(ctx: CrawlerContext, num_workers: int = 10) -> None:
    workers = [asyncio.create_task(_worker(ctx)) for _ in range(num_workers)]
    try:
        while True:
            await ctx.queue.join()
            pending = [task for task in ctx.background_tasks if not task.done()]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)
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
        init_db()
        await async_init_db()
        async with db:
            async with make_page_client() as client, make_aux_client() as aux_client:
                with Live(display, refresh_per_second=4):
                    task_id = progress.add_task("Crawling...", total=None)

                    ctx = CrawlerContext(
                        progress=progress,
                        task_id=task_id,
                        client=client,
                        aux_client=aux_client,
                        stats=stats,
                    )

                    for url in SEED_URLS:
                        await _enqueue_url(normalize_url(url), ctx)

                    try:
                        await _crawl(ctx)
                    except KeyboardInterrupt:
                        progress.print("[yellow]Shutting down...[/yellow]")
                        ctx.shutdown_event.set()

                    progress.update(task_id, description="Crawling complete")

            rprint()
            rprint(stats.summary())

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
