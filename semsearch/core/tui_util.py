import threading
from dataclasses import dataclass, field

from rich.console import Group
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.text import Text


class SpeedColumn(ProgressColumn):
    def __init__(self, unit: str = "doc/s") -> None:
        self._unit = unit
        super().__init__()

    def render(self, task):
        if task.speed is None:
            return Text(f"? {self._unit}")
        return Text(f"{task.speed:.1f} {self._unit}")


def make_determinate_progress(unit: str = "doc/s") -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.percentage:>3.0f}%"),
        SpeedColumn(unit),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def make_indeterminate_progress(
    count_text: str = "{task.completed}", unit: str = "doc/s"
) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn(count_text),
        SpeedColumn(unit),
        TimeElapsedColumn(),
    )


@dataclass
class CrawlStats:
    # crawler engine
    in_flight: int = 0
    rate_limited: int = 0
    robots_blocked: int = 0
    # network / http
    requests: int = 0
    req_2xx: int = 0
    req_3xx: int = 0
    req_4xx: int = 0
    req_5xx: int = 0
    error_net: int = 0
    # crawl data
    visited: int = 0
    saved: int = 0
    skipped: int = 0
    sitemap_urls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, name: str, by: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + by)

    def __rich__(self) -> Group:
        def row(pairs: list[tuple[str, str, str]]) -> Text:
            t = Text()
            for label, value, style in pairs:
                t.append(f" {label} ", style="dim")
                t.append(value, style=style)
            return t

        engine_row = row(
            [
                ("in-flight", f"{self.in_flight:,}", "bold white"),
                ("rate-limited", f"{self.rate_limited:,}", "yellow"),
                ("robots-blocked", f"{self.robots_blocked:,}", "bold yellow"),
            ]
        )
        network_row = row(
            [
                ("requests", f"{self.requests:,}", "white"),
                ("2xx", f"{self.req_2xx:,}", "bold green"),
                ("3xx", f"{self.req_3xx:,}", "cyan"),
                ("4xx", f"{self.req_4xx:,}", "bold red"),
                ("5xx", f"{self.req_5xx:,}", "bold magenta"),
                ("net-err", f"{self.error_net:,}", "red"),
            ]
        )
        data_row = row(
            [
                ("discovered", f"{self.visited:,}", "bold white"),
                ("saved", f"{self.saved:,}", "bold green"),
                ("skipped", f"{self.skipped:,}", "cyan"),
                ("sitemap-urls", f"{self.sitemap_urls:,}", "cyan"),
            ]
        )
        return Group(engine_row, network_row, data_row)


def make_crawler_display(progress: Progress, stats: CrawlStats) -> Group:
    """Combine progress bar and stats lines into a single Live-compatible renderable."""
    return Group(progress, Rule(style="dim"), stats)
