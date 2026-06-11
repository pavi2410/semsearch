import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .storage import init_db, save_page
from .storage.models import Block

OLD_WEBPAGES_DIR = Path("data") / "webpages"  # pre-v2 flat JSON
OLD_PAGES_DIR = Path("data") / "pages"  # v2 per-hostname JSON dirs
OLD_BLOCKS_FILE = Path("data") / "blocks.json"  # v2 blocks JSON


def _migrate_webpages(console: Console) -> int:
    """Migrate pre-v2 flat webpages/ directory."""
    if not OLD_WEBPAGES_DIR.is_dir():
        return 0

    files = sorted(OLD_WEBPAGES_DIR.glob("*.json"))
    console.print(f"Migrating [bold]{len(files)}[/bold] files from {OLD_WEBPAGES_DIR}")

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        url: str = data["url"]
        html: str = data["content"]
        last_fetched_ms = data.get("lastFetchedAt", 0)
        last_fetched = datetime.fromtimestamp(
            last_fetched_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        save_page(url, html, last_fetched)
        fp.unlink()

    shutil.rmtree(OLD_WEBPAGES_DIR, ignore_errors=True)
    return len(files)


def _migrate_pages(console: Console) -> int:
    """Migrate v2 per-hostname JSON page metadata into SQLite."""
    if not OLD_PAGES_DIR.is_dir():
        return 0

    files = sorted(OLD_PAGES_DIR.rglob("*.json"))
    console.print(
        f"Migrating [bold]{len(files)}[/bold] page metadata files from {OLD_PAGES_DIR}"
    )

    from .storage.models import SyncPage as Page

    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "url_hash": data.get("urlHash") or _url_hash(data["url"]),
                "url": data["url"],
                "fetched_at": data.get("lastFetchedAt", ""),
                "content_hash": data.get("contentHash", ""),
                "title": data.get("title"),
            }
        )

    # Bulk insert in chunks
    chunk = 500
    for i in range(0, len(rows), chunk):
        Page.replace_many(rows[i : i + chunk]).execute()

    shutil.rmtree(OLD_PAGES_DIR, ignore_errors=True)
    return len(files)


def _migrate_blocks(console: Console) -> int:
    """Migrate v2 blocks.json into SQLite blocks table."""
    if not OLD_BLOCKS_FILE.exists():
        return 0

    try:
        data = json.loads(OLD_BLOCKS_FILE.read_text())
    except Exception:
        console.print(f"[yellow]Could not parse {OLD_BLOCKS_FILE}, skipping[/yellow]")
        return 0

    rows = []
    for domain, entry in data.get("domains", {}).items():
        rows.append(
            {
                "key": domain,
                "kind": "domain",
                "reason": entry.get("reason", ""),
                "permanent": entry.get("permanent", True),
                "until": entry.get("until"),
            }
        )
    for url in data.get("urls", []):
        rows.append(
            {
                "key": url,
                "kind": "url",
                "reason": "migrated",
                "permanent": True,
                "until": None,
            }
        )

    if rows:
        Block.replace_many(rows).execute()

    OLD_BLOCKS_FILE.unlink()
    console.print(f"Migrated [bold]{len(rows)}[/bold] block entries")
    return len(rows)


def _url_hash(url: str) -> str:
    import hashlib
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    parsed = parsed._replace(
        fragment="", scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()
    )
    path = parsed.path.rstrip("/") if len(parsed.path) > 1 else parsed.path
    parsed = parsed._replace(path=path)
    return hashlib.sha256(urlunparse(parsed).encode()).hexdigest()[:16]


def main() -> None:
    console = Console()
    init_db()

    total = 0
    total += _migrate_webpages(console)
    total += _migrate_pages(console)
    _migrate_blocks(console)

    if total == 0:
        console.print("[dim]Nothing to migrate — already up to date[/dim]")
    else:
        console.print(f"[green]Migration complete — {total} pages migrated[/green]")
