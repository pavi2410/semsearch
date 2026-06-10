import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_BLOCKS_FILE = Path("data") / "blocks.json"
_5XX_THRESHOLD = 3


@dataclass
class BlockEntry:
    reason: str
    permanent: bool
    until: float | None  # epoch seconds; None means permanent


class BlockList:
    """Tracks blocked domains and URLs. Domain blocks are persisted; 5xx fail counts are session-only."""

    def __init__(self) -> None:
        self._blocked_domains: dict[str, BlockEntry] = {}
        self._blocked_urls: set[str] = set()
        self._fail_counts: dict[
            str, int
        ] = {}  # URL → consecutive 5xx count, session-only
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_blocked(self, url: str) -> tuple[bool, str]:
        """Return (blocked, reason). Expired temporary blocks are auto-cleared."""
        if url in self._blocked_urls:
            return True, "url"

        from urllib.parse import urlparse

        domain = urlparse(url).hostname or ""
        entry = self._blocked_domains.get(domain)
        if entry is None:
            return False, ""

        if (
            not entry.permanent
            and entry.until is not None
            and time.time() >= entry.until
        ):
            del self._blocked_domains[domain]
            self._save()
            return False, ""

        return True, f"domain:{entry.reason}"

    def record(self, url: str, status_code: int, retry_after: float | None) -> None:
        """Record a failed response and block as appropriate."""
        from urllib.parse import urlparse

        domain = urlparse(url).hostname or ""

        if status_code in (403, 451):
            self._blocked_domains[domain] = BlockEntry(
                reason=str(status_code), permanent=True, until=None
            )
            self._save()

        elif status_code == 429:
            wait = retry_after if retry_after is not None else 60.0
            self._blocked_domains[domain] = BlockEntry(
                reason="429", permanent=False, until=time.time() + wait
            )
            self._save()

        elif status_code == 404 or status_code == 410:
            self._blocked_urls.add(url)
            self._save()

        elif 500 <= status_code < 600:
            count = self._fail_counts.get(url, 0) + 1
            self._fail_counts[url] = count
            if count >= _5XX_THRESHOLD:
                self._blocked_urls.add(url)
                self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not _BLOCKS_FILE.exists():
            return
        try:
            data = json.loads(_BLOCKS_FILE.read_text())
            for domain, entry in data.get("domains", {}).items():
                self._blocked_domains[domain] = BlockEntry(**entry)
            self._blocked_urls = set(data.get("urls", []))
        except Exception:
            pass

    def _save(self) -> None:
        _BLOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "domains": {d: asdict(e) for d, e in self._blocked_domains.items()},
            "urls": list(self._blocked_urls),
        }
        _BLOCKS_FILE.write_text(json.dumps(data, indent=2))
