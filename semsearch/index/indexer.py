import argparse
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from rich.console import Console

from ..core.tui_util import make_determinate_progress
from ..crawl.content_filter import is_indexable_page
from ..crawl.metadata import PageMetadata, extract_page_metadata
from ..search.index_store import dump_index, load_previous_doc_ids
from ..search.ranking import compute_pagerank_boosts
from ..storage import init_db, iter_page_metas, read_content, url_hash
from ..storage.embedding_cache import load_embedding, save_embedding
from ..storage.models import SyncLink as Link
from ..storage.models import SyncPage as Page
from ..storage.page import normalize_url
from ..storage.token_cache import load_tokens, save_tokens
from .embeddings import DEFAULT_MODEL, DocumentEmbedding, build_embedding_index, embed_document
from .nlp import preprocess


@dataclass
class IndexPlan:
    to_process: list[dict]
    reused: dict[str, list[str]]


@dataclass
class IndexStats:
    total_pages: int
    changed: int
    reused: int
    removed: int
    skipped: int


def plan_index(
    pages: list[dict],
    *,
    force: bool,
    token_loader: Callable[[str], list[str] | None] = load_tokens,
) -> IndexPlan:
    to_process: list[dict] = []
    reused: dict[str, list[str]] = {}

    for page in pages:
        content_hash = page["contentHash"]
        if not force and page.get("indexedContentHash") == content_hash:
            tokens = token_loader(content_hash)
            if tokens is not None:
                reused[page["urlHash"]] = tokens
                continue
        to_process.append(page)

    return IndexPlan(to_process=to_process, reused=reused)


def build_index_stats(
    pages: list[dict], plan: IndexPlan, *, previous_doc_ids: list[str], skipped: int
) -> IndexStats:
    current_doc_ids = {page["urlHash"] for page in pages}
    removed = len(set(previous_doc_ids) - current_doc_ids)
    return IndexStats(
        total_pages=len(pages),
        changed=len(plan.to_process),
        reused=len(plan.reused),
        removed=removed,
        skipped=skipped,
    )


def _process_page(meta: dict) -> tuple[str, str, str, PageMetadata, list[str]] | None:
    url = meta["url"]
    content_hash = meta["contentHash"]
    html = read_content(content_hash)
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
    return url, doc_id, content_hash, page_meta, tokens


def _page_update_fields(page_meta: PageMetadata, content_hash: str) -> dict[str, str]:
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
        "language": page_meta.language,
        "indexed_content_hash": content_hash,
    }


def _save_links(doc_id: str, outbound_links: list[str]) -> None:
    Link.delete().where(Link.source_hash == doc_id).execute()
    if not outbound_links:
        return
    Link.insert_many(
        [{"source_hash": doc_id, "target_url": target} for target in outbound_links]
    ).execute()


def _persist_page(doc_id: str, content_hash: str, page_meta: PageMetadata) -> None:
    Page.update(**_page_update_fields(page_meta, content_hash)).where(
        Page.url_hash == doc_id
    ).execute()
    _save_links(doc_id, page_meta.outbound_links)


def _clear_index_state(doc_id: str) -> None:
    Page.update(indexed_content_hash=None).where(Page.url_hash == doc_id).execute()


def _build_pagerank(doc_ids: list[str]) -> dict[str, float]:
    doc_id_set = set(doc_ids)
    url_to_doc = {
        normalize_url(page.url): page.url_hash
        for page in Page.select(Page.url, Page.url_hash)
        if page.url_hash in doc_id_set
    }
    links = [(link.source_hash, link.target_url) for link in Link.select()]
    return compute_pagerank_boosts(doc_ids, url_to_doc, links)


def _build_embeddings(
    doc_ids: list[str],
    *,
    force: bool,
    console: Console,
):
    embedder = TextEmbedding(model_name=DEFAULT_MODEL)
    doc_embeddings: dict[str, DocumentEmbedding] = {}
    cached = 0
    embedded = 0

    for doc_id in doc_ids:
        page = Page.get_by_id(doc_id)
        if not force:
            cached_embedding = load_embedding(page.content_hash)
            if cached_embedding is not None:
                chunks, vectors = cached_embedding
                doc_embeddings[doc_id] = DocumentEmbedding(chunks=chunks, vectors=vectors)
                cached += 1
                continue

        html = read_content(page.content_hash)
        if not is_indexable_page(page.url, html):
            continue
        page_meta = extract_page_metadata(html, page.url)
        doc_embedding = embed_document(page_meta, model=embedder)
        if doc_embedding is None:
            continue
        save_embedding(page.content_hash, doc_embedding.chunks, doc_embedding.vectors)
        doc_embeddings[doc_id] = doc_embedding
        embedded += 1

    console.print(f"[dim]{cached} embedding cache hits, {embedded} embedded[/dim]")
    return build_embedding_index(doc_ids, doc_embeddings)


def _run_process_pool(
    to_process: list[dict],
    entries: dict[str, list[str]],
    progress,
    task,
    console: Console,
) -> tuple[int, bool]:
    skipped = 0
    interrupted = False

    pool = ProcessPoolExecutor(max_workers=os.cpu_count())
    try:
        future_to_meta = {
            pool.submit(_process_page, meta): meta for meta in to_process
        }
        try:
            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                result = future.result()
                progress.advance(task)
                if result is None:
                    skipped += 1
                    _clear_index_state(meta["urlHash"])
                    continue
                url, doc_id, content_hash, page_meta, tokens = result
                entries[doc_id] = tokens
                save_tokens(content_hash, tokens)
                _persist_page(doc_id, content_hash, page_meta)
        except KeyboardInterrupt:
            console.print("[yellow]Shutting down...[/yellow]")
            interrupted = True
            for future in future_to_meta:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
    except BrokenProcessPool:
        console.print(
            "[yellow]ProcessPoolExecutor failed, falling back to sequential indexing[/yellow]"
        )
        for meta in to_process:
            result = _process_page(meta)
            progress.advance(task)
            if result is None:
                skipped += 1
                _clear_index_state(meta["urlHash"])
                continue
            url, doc_id, content_hash, page_meta, tokens = result
            entries[doc_id] = tokens
            save_tokens(content_hash, tokens)
            _persist_page(doc_id, content_hash, page_meta)
    else:
        pool.shutdown()

    return skipped, interrupted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the BM25 search index")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all pages, ignoring cached tokens",
    )
    args = parser.parse_args(argv)

    init_db()
    pages = list(iter_page_metas())
    domains = Counter(page["url"].split("/")[2] for page in pages)
    plan = plan_index(pages, force=args.force)
    previous_doc_ids = load_previous_doc_ids()

    console = Console()
    console.print(
        f"Found [bold]{len(pages)}[/bold] webpages"
        f" from [bold]{len(domains)}[/bold] unique domains"
    )
    if args.force:
        console.print("[dim]Force mode — reprocessing all pages[/dim]")

    entries: dict[str, list[str]] = dict(plan.reused)
    skipped = 0
    interrupted = False

    progress = make_determinate_progress()

    with progress:
        if plan.to_process:
            task = progress.add_task("Indexing changed pages", total=len(plan.to_process))
            skipped, interrupted = _run_process_pool(
                plan.to_process, entries, progress, task, console
            )
        else:
            console.print("[dim]No changed pages — reusing cached tokens[/dim]")

    stats = build_index_stats(pages, plan, previous_doc_ids=previous_doc_ids, skipped=skipped)
    console.print(
        f"[dim]{stats.reused} reused, {stats.changed} changed"
        f"{f', {stats.removed} removed' if stats.removed else ''}"
        f"{f', {stats.skipped} skipped' if stats.skipped else ''}[/dim]"
    )

    if interrupted:
        console.print(f"Interrupted — [bold]{len(entries)}[/bold] pages indexed so far")
        return

    if not entries:
        console.print("[yellow]No indexable pages found[/yellow]")
        return

    doc_ids = list(entries.keys())
    corpus_tokens = [entries[doc_id] for doc_id in doc_ids]
    bm25 = BM25Okapi(corpus_tokens)
    pagerank = _build_pagerank(doc_ids)
    embedding_index = _build_embeddings(doc_ids, force=args.force, console=console)
    dump_index(bm25, doc_ids, pagerank, embedding_index)
    console.print(
        f"[green]Built index with {len(doc_ids)} documents"
        f", PageRank boosts"
        f"{', and semantic embeddings' if embedding_index else ''}[/green]"
    )


if __name__ == "__main__":
    main()
