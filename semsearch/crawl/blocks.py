import time
from urllib.parse import urlparse

from ..storage.models import Block


_5XX_THRESHOLD = 3


class BlockList:
    """Tracks blocked domains and URLs. Domain blocks are persisted via SQLite; 5xx fail counts are session-only."""

    def __init__(self) -> None:
        self._fail_counts: dict[str, int] = {}

    def is_blocked(self, url: str) -> tuple[bool, str]:
        """Return (blocked, reason). Expired temporary blocks are auto-cleared."""
        try:
            Block.get((Block.key == url) & (Block.kind == "url"))
            return True, "url"
        except Block.DoesNotExist:
            pass

        domain = urlparse(url).hostname or ""
        try:
            block = Block.get((Block.key == domain) & (Block.kind == "domain"))
        except Block.DoesNotExist:
            return False, ""

        if (
            not block.permanent
            and block.until is not None
            and time.time() >= block.until
        ):
            Block.delete().where(Block.key == domain).execute()
            return False, ""

        return True, f"domain:{block.reason}"

    def record(self, url: str, status_code: int, retry_after: float | None) -> None:
        """Record a failed response and block as appropriate."""
        domain = urlparse(url).hostname or ""

        if status_code in (403, 451):
            Block.replace(
                key=domain,
                kind="domain",
                reason=str(status_code),
                permanent=True,
                until=None,
            ).execute()

        elif status_code == 429:
            wait = retry_after if retry_after is not None else 60.0
            Block.replace(
                key=domain,
                kind="domain",
                reason="429",
                permanent=False,
                until=time.time() + wait,
            ).execute()

        elif status_code in (404, 410):
            Block.replace(
                key=url,
                kind="url",
                reason=str(status_code),
                permanent=True,
                until=None,
            ).execute()

        elif 500 <= status_code < 600:
            count = self._fail_counts.get(url, 0) + 1
            self._fail_counts[url] = count
            if count >= _5XX_THRESHOLD:
                Block.replace(
                    key=url,
                    kind="url",
                    reason="5xx",
                    permanent=True,
                    until=None,
                ).execute()
