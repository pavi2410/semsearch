import hashlib
import json
from collections import Counter

from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from .core.config import WEBPAGES_DIR
from .core.html_utils import extract_metadata
from .core.index_store import dump_docs, dump_index
from .core.nlp import preprocess


class SpeedColumn(ProgressColumn):
    def render(self, task):
        if task.speed is None:
            return Text("? doc/s")
        return Text(f"{task.speed:.1f} doc/s")


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def main() -> None:
    files = list(WEBPAGES_DIR.glob("*.json"))

    domains = Counter(fp.stem.rsplit("_", 1)[0] for fp in files)
    console = Console()
    console.print(
        f"Found [bold]{len(files)}[/bold] webpages"
        f" from [bold]{len(domains)}[/bold] unique domains"
    )

    docs: dict[str, dict[str, str]] = {}
    entries: list[tuple[str, str, list[str]]] = []

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.percentage:>3.0f}%"),
        SpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task("Indexing", total=len(files))
        for fp in files:
            progress.update(task, advance=1)
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)

            url: str = data["url"]
            html: str = data["content"]
            title, text = extract_metadata(html)

            doc_id = _url_hash(url)
            docs[doc_id] = {"url": url, "title": title}
            tokens = preprocess(f"{title} {text}")
            entries.append((fp.name, doc_id, tokens))

    entries.sort(key=lambda x: x[0])
    doc_ids = [e[1] for e in entries]
    corpus_tokens = [e[2] for e in entries]

    bm25 = BM25Okapi(corpus_tokens)

    dump_index(bm25, doc_ids)
    dump_docs(docs)


if __name__ == "__main__":
    main()
