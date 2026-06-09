import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from rank_bm25 import BM25Okapi
from rich.console import Console

from .core.html_utils import extract_metadata
from .core.index_store import dump_docs, dump_index
from .core.nlp import preprocess
from .core.tui_util import make_determinate_progress
from .storage import iter_page_metas, read_content, url_hash


def _process_page(meta: dict) -> tuple[str, str, str, list[str]]:
    url = meta["url"]
    html = read_content(meta["contentHash"])
    title, text = extract_metadata(html)
    doc_id = url_hash(url)
    tokens = preprocess(f"{title} {text}")
    return (meta["url"], doc_id, title, tokens)


def main() -> None:
    metas = list(iter_page_metas())
    domains = Counter(meta["url"].split("/")[2] for meta in metas)
    console = Console()
    console.print(
        f"Found [bold]{len(metas)}[/bold] webpages"
        f" from [bold]{len(domains)}[/bold] unique domains"
    )

    docs: dict[str, dict[str, str]] = {}
    entries: dict[str, tuple[str, list[str]]] = {}
    interrupted = False

    progress = make_determinate_progress()

    with progress:
        task = progress.add_task("Indexing", total=len(metas))

        pool = ProcessPoolExecutor(max_workers=os.cpu_count())
        try:
            futures = [pool.submit(_process_page, meta) for meta in metas]
            try:
                for future in as_completed(futures):
                    url, doc_id, title, tokens = future.result()
                    docs[doc_id] = {"url": url, "title": title}
                    entries[doc_id] = (url, tokens)
                    progress.advance(task)
            except KeyboardInterrupt:
                console.print("[yellow]Shutting down...[/yellow]")
                interrupted = True
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
        except BrokenProcessPool:
            console.print("[yellow]ProcessPoolExecutor failed, falling back to sequential indexing[/yellow]")
            for meta in metas:
                url, doc_id, title, tokens = _process_page(meta)
                docs[doc_id] = {"url": url, "title": title}
                entries[doc_id] = (url, tokens)
                progress.advance(task)
        else:
            pool.shutdown()

    if interrupted:
        console.print(f"Interrupted — [bold]{len(entries)}[/bold] pages indexed so far")
        return

    doc_ids = list(entries.keys())
    corpus_tokens = [tokens for _, tokens in entries.values()]

    bm25 = BM25Okapi(corpus_tokens)

    dump_index(bm25, doc_ids)
    dump_docs(docs)


if __name__ == "__main__":
    main()
