from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
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
