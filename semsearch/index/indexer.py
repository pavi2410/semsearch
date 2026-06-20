import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from rank_bm25 import BM25Okapi
from rich.console import Console

from ..core.tui_util import make_determinate_progress
from ..crawl.content_filter import is_indexable_page
from ..crawl.metadata import PageMetadata, extract_page_metadata
from ..search.index_store import dump_index
from ..storage import init_db, iter_page_metas, read_content, url_hash
from ..storage.models import SyncLink as Link
from ..storage.models import SyncPage as Page
from .nlp import preprocess


def _process_page(meta: dict) -> tuple[str, str, PageMetadata, list[str]] | None:
    url = meta["url"]
    html = read_content(meta["contentHash"])
    if not is_indexable_page(url, html):
        return None
    page_meta = extract_page_metadata(html, url)
    doc_id = url_hash(url)
    index_text = " ".join(
        part
        for part in (page_meta.title, page_meta.description, page_meta.body_text)
        if part
    )
    tokens = preprocess(index_text)
    return url, doc_id, page_meta, tokens


def _page_update_fields(page_meta: PageMetadata) -> dict[str, str]:
    return {
        "title": page_meta.title,
        "description": page_meta.description,
        "canonical_url": page_meta.canonical_url,
        "og_title": page_meta.og_title,
        "og_description": page_meta.og_description,
        "published_at": page_meta.published_at,
        "modified_at": page_meta.modified_at,
        "body_excerpt": page_meta.body_excerpt,
        "jsonld_types": ",".join(page_meta.jsonld_types),
    }


def _save_links(doc_id: str, outbound_links: list[str]) -> None:
    Link.delete().where(Link.source_hash == doc_id).execute()
    if not outbound_links:
        return
    Link.insert_many(
        [{"source_hash": doc_id, "target_url": target} for target in outbound_links]
    ).execute()


def _persist_page(doc_id: str, page_meta: PageMetadata) -> None:
    Page.update(**_page_update_fields(page_meta)).where(Page.url_hash == doc_id).execute()
    _save_links(doc_id, page_meta.outbound_links)


def main() -> None:
    init_db()
    metas = list(iter_page_metas())
    domains = Counter(meta["url"].split("/")[2] for meta in metas)
    console = Console()
    console.print(
        f"Found [bold]{len(metas)}[/bold] webpages"
        f" from [bold]{len(domains)}[/bold] unique domains"
    )

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
                    result = future.result()
                    progress.advance(task)
                    if result is None:
                        continue
                    url, doc_id, page_meta, tokens = result
                    entries[doc_id] = (url, tokens)
                    _persist_page(doc_id, page_meta)
            except KeyboardInterrupt:
                console.print("[yellow]Shutting down...[/yellow]")
                interrupted = True
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
        except BrokenProcessPool:
            console.print(
                "[yellow]ProcessPoolExecutor failed, falling back to sequential indexing[/yellow]"
            )
            for meta in metas:
                result = _process_page(meta)
                progress.advance(task)
                if result is None:
                    continue
                url, doc_id, page_meta, tokens = result
                entries[doc_id] = (url, tokens)
                _persist_page(doc_id, page_meta)
        else:
            pool.shutdown()

    if interrupted:
        console.print(f"Interrupted — [bold]{len(entries)}[/bold] pages indexed so far")
        return

    doc_ids = list(entries.keys())
    corpus_tokens = [tokens for _, tokens in entries.values()]

    bm25 = BM25Okapi(corpus_tokens)

    dump_index(bm25, doc_ids)


if __name__ == "__main__":
    main()
