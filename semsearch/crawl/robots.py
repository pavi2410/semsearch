import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "semsearch"


class RobotsCache:
    """Fetches and caches robots.txt per domain. Async, single-event-loop safe."""

    def __init__(self, client: httpx.AsyncClient, default_delay: float = 1.0) -> None:
        self._client = client
        self._default_delay = default_delay
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def _get_parser(self, domain: str) -> RobotFileParser:
        if domain in self._parsers:
            return self._parsers[domain]

        async with self._get_lock(domain):
            # Re-check after acquiring lock
            if domain in self._parsers:
                return self._parsers[domain]

            parser = RobotFileParser()
            robots_url = f"https://{domain}/robots.txt"
            try:
                resp = await self._client.get(
                    robots_url, timeout=5, follow_redirects=True
                )
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    parser.allow_all = True  # non-200 → treat as allow all
            except Exception:
                parser.allow_all = True  # network error → allow all

            self._parsers[domain] = parser

        return parser

    async def can_fetch(self, url: str) -> bool:
        domain = urlparse(url).hostname
        if not domain:
            return True
        parser = await self._get_parser(domain)
        return parser.can_fetch(USER_AGENT, url)

    async def sitemaps(self, domain: str) -> list[str]:
        """Returns sitemap URLs declared in robots.txt for this domain."""
        parser = await self._get_parser(domain)
        return list(parser.site_maps() or [])

    async def crawl_delay(self, url: str) -> float:
        """Returns the effective crawl delay for this URL's domain.

        Uses the robots.txt Crawl-delay if specified, otherwise the default.
        """
        domain = urlparse(url).hostname
        if not domain:
            return self._default_delay
        parser = await self._get_parser(domain)
        delay = parser.crawl_delay(USER_AGENT)
        if delay is not None:
            return max(self._default_delay, float(delay))
        return self._default_delay
