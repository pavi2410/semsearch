import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .storage import save_page

OLD_DIR = Path("data") / "webpages"


def main() -> None:
    console = Console()

    if not OLD_DIR.is_dir():
        console.print("[red]No old data/webpages/ directory found[/red]")
        return

    files = sorted(OLD_DIR.glob("*.json"))
    console.print(f"Migrating [bold]{len(files)}[/bold] files from {OLD_DIR}")

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

    shutil.rmtree(OLD_DIR, ignore_errors=True)
    console.print("[green]Migration complete[/green]")
