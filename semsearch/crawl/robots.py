import threading
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "semsearch"


class RobotsCache:
    """Fetches and caches robots.txt per domain. Thread-safe."""

    def __init__(self, client: httpx.Client, default_delay: float = 1.0) -> None:
        self._client = client
        self._default_delay = default_delay
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def _get_lock(self, domain: str) -> threading.Lock:
        with self._meta_lock:
            if domain not in self._locks:
                self._locks[domain] = threading.Lock()
            return self._locks[domain]

    def _get_parser(self, domain: str) -> RobotFileParser:
        # Fast path: already cached
        with self._meta_lock:
            if domain in self._parsers:
                return self._parsers[domain]

        # Slow path: fetch with per-domain lock to avoid races
        lock = self._get_lock(domain)
        with lock:
            # Re-check after acquiring lock
            with self._meta_lock:
                if domain in self._parsers:
                    return self._parsers[domain]

            parser = RobotFileParser()
            robots_url = f"https://{domain}/robots.txt"
            try:
                resp = self._client.get(robots_url, timeout=5, follow_redirects=True)
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                # Any non-200 (404, 403, etc.) → treat as allow all, parser stays empty
            except Exception:
                pass  # Network error → allow all

            with self._meta_lock:
                self._parsers[domain] = parser

        return parser

    def can_fetch(self, url: str) -> bool:
        domain = urlparse(url).hostname
        if not domain:
            return True
        return self._get_parser(domain).can_fetch(USER_AGENT, url)

    def crawl_delay(self, url: str) -> float:
        """Returns the effective crawl delay for this URL's domain.

        Uses the robots.txt Crawl-delay if specified, otherwise the default.
        """
        domain = urlparse(url).hostname
        if not domain:
            return self._default_delay
        delay = self._get_parser(domain).crawl_delay(USER_AGENT)
        if delay is not None:
            return max(self._default_delay, float(delay))
        return self._default_delay
