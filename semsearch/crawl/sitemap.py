import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx

from .robots import USER_AGENT

_NS_SITEMAP = "http://www.sitemaps.org/schemas/sitemap/0.9"
_HEADERS = {"User-Agent": USER_AGENT}
_MAX_DEPTH = 3  # maximum sitemap index recursion depth


class SitemapLoader:
    """Fetches and parses sitemaps for a domain. Async."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def load(self, domain: str, sitemap_urls: list[str]) -> list[str]:
        """Return all page URLs found across the given sitemap URLs for a domain.

        Falls back to /sitemap.xml if no sitemap URLs are provided.
        """
        if not sitemap_urls:
            sitemap_urls = [f"https://{domain}/sitemap.xml"]

        urls: list[str] = []
        seen_sitemaps: set[str] = set()
        for sitemap_url in sitemap_urls:
            await self._fetch(sitemap_url, urls, seen_sitemaps, depth=0)
        return urls

    async def _fetch(
        self,
        sitemap_url: str,
        urls: list[str],
        seen_sitemaps: set[str],
        depth: int,
    ) -> None:
        if depth > _MAX_DEPTH or sitemap_url in seen_sitemaps:
            return
        seen_sitemaps.add(sitemap_url)

        try:
            resp = await self._client.get(
                sitemap_url, headers=_HEADERS, follow_redirects=True, timeout=10
            )
            if resp.status_code != 200:
                return
            root = ET.fromstring(resp.text)
        except Exception:
            return

        tag = root.tag.removeprefix(f"{{{_NS_SITEMAP}}}")

        if tag == "sitemapindex":
            # Recurse into each child sitemap
            for sitemap_el in root.findall(f"{{{_NS_SITEMAP}}}sitemap"):
                loc = sitemap_el.findtext(f"{{{_NS_SITEMAP}}}loc")
                if loc:
                    await self._fetch(loc.strip(), urls, seen_sitemaps, depth + 1)

        elif tag == "urlset":
            for url_el in root.findall(f"{{{_NS_SITEMAP}}}url"):
                loc = url_el.findtext(f"{{{_NS_SITEMAP}}}loc")
                if loc:
                    parsed = urlparse(loc.strip())
                    if parsed.scheme in ("http", "https"):
                        urls.append(loc.strip())
