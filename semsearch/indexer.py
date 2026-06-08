from collections import Counter

from rank_bm25 import BM25Okapi
from rich.console import Console

from .core.html_utils import extract_metadata
from .core.index_store import dump_docs, dump_index
from .core.nlp import preprocess
from .core.tui_util import make_determinate_progress
from .storage import iter_page_metas, read_content, url_hash


def main() -> None:
    metas = list(iter_page_metas())
    domains = Counter(meta["url"].split("/")[2] for meta in metas)
    console = Console()
    console.print(
        f"Found [bold]{len(metas)}[/bold] webpages"
        f" from [bold]{len(domains)}[/bold] unique domains"
    )

    docs: dict[str, dict[str, str]] = {}
    entries: list[tuple[str, str, list[str]]] = []

    progress = make_determinate_progress()

    with progress:
        task = progress.add_task("Indexing", total=len(metas))
        for meta in metas:
            progress.update(task, advance=1)
            url: str = meta["url"]
            content_hash: str = meta["contentHash"]

            html = read_content(content_hash)
            title, text = extract_metadata(html)

            doc_id = url_hash(url)
            docs[doc_id] = {"url": url, "title": title}
            tokens = preprocess(f"{title} {text}")
            entries.append((meta["url"], doc_id, tokens))

    entries.sort(key=lambda x: x[0])
    doc_ids = [e[1] for e in entries]
    corpus_tokens = [e[2] for e in entries]

    bm25 = BM25Okapi(corpus_tokens)

    dump_index(bm25, doc_ids)
    dump_docs(docs)


if __name__ == "__main__":
    main()
