import httpx

from semsearch.crawl.robots import USER_AGENT, RobotsCache

ROBOTS_ALLOW_ALL = ""
ROBOTS_DISALLOW_ALL = f"User-agent: {USER_AGENT}\nDisallow: /\n"
ROBOTS_DISALLOW_ADMIN = f"User-agent: {USER_AGENT}\nDisallow: /admin/\n"
ROBOTS_CRAWL_DELAY = f"User-agent: {USER_AGENT}\nDisallow:\nCrawl-delay: 5\n"
ROBOTS_WITH_SITEMAP = (
    "User-agent: *\nDisallow:\nSitemap: https://example.com/sitemap.xml\n"
)


def make_client(responses: dict[str, tuple[int, str]]) -> httpx.AsyncClient:
    """Build an AsyncClient with a mock transport mapping URL → (status, body)."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, (status, body) in responses.items():
            if url.startswith(pattern):
                return httpx.Response(status, text=body)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------
# can_fetch
# ------------------------------------------------------------------


async def test_can_fetch_allow_all():
    client = make_client({"https://example.com/robots.txt": (200, ROBOTS_ALLOW_ALL)})
    cache = RobotsCache(client)
    assert await cache.can_fetch("https://example.com/page") is True


async def test_can_fetch_disallow_all():
    client = make_client({"https://example.com/robots.txt": (200, ROBOTS_DISALLOW_ALL)})
    cache = RobotsCache(client)
    assert await cache.can_fetch("https://example.com/page") is False


async def test_can_fetch_disallow_path():
    client = make_client(
        {"https://example.com/robots.txt": (200, ROBOTS_DISALLOW_ADMIN)}
    )
    cache = RobotsCache(client)
    assert await cache.can_fetch("https://example.com/admin/secret") is False
    assert await cache.can_fetch("https://example.com/public") is True


async def test_can_fetch_robots_404_allows_all():
    client = make_client({})  # 404 for everything
    cache = RobotsCache(client)
    assert await cache.can_fetch("https://example.com/anything") is True


async def test_can_fetch_network_error_allows_all():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cache = RobotsCache(client)
    assert await cache.can_fetch("https://example.com/page") is True


async def test_robots_fetched_once_per_domain():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, text=ROBOTS_ALLOW_ALL)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cache = RobotsCache(client)
    await cache.can_fetch("https://example.com/a")
    await cache.can_fetch("https://example.com/b")
    await cache.can_fetch("https://example.com/c")
    assert call_count == 1


# ------------------------------------------------------------------
# crawl_delay
# ------------------------------------------------------------------


async def test_crawl_delay_from_robots():
    client = make_client({"https://example.com/robots.txt": (200, ROBOTS_CRAWL_DELAY)})
    cache = RobotsCache(client, default_delay=1.0)
    delay = await cache.crawl_delay("https://example.com/page")
    assert delay == 5.0


async def test_crawl_delay_uses_default_when_not_set():
    client = make_client({"https://example.com/robots.txt": (200, ROBOTS_ALLOW_ALL)})
    cache = RobotsCache(client, default_delay=2.0)
    delay = await cache.crawl_delay("https://example.com/page")
    assert delay == 2.0


async def test_crawl_delay_respects_minimum():
    # robots.txt says 0.5s but default is 1.0 — should use max
    robots = f"User-agent: {USER_AGENT}\nDisallow:\nCrawl-delay: 0.5\n"
    client = make_client({"https://example.com/robots.txt": (200, robots)})
    cache = RobotsCache(client, default_delay=1.0)
    delay = await cache.crawl_delay("https://example.com/page")
    assert delay == 1.0


# ------------------------------------------------------------------
# sitemaps
# ------------------------------------------------------------------


async def test_sitemaps_from_robots():
    client = make_client({"https://example.com/robots.txt": (200, ROBOTS_WITH_SITEMAP)})
    cache = RobotsCache(client)
    sitemaps = await cache.sitemaps("example.com")
    assert "https://example.com/sitemap.xml" in sitemaps


async def test_sitemaps_empty_when_none_declared():
    client = make_client({"https://example.com/robots.txt": (200, ROBOTS_ALLOW_ALL)})
    cache = RobotsCache(client)
    sitemaps = await cache.sitemaps("example.com")
    assert sitemaps == []
