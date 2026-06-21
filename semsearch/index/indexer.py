import argparse
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from rich.console import Console

from fastembed import TextEmbedding

from ..core.tui_util import make_determinate_progress, run_with_progress_refresh
from ..crawl.content_filter import is_indexable_page
from ..crawl.metadata import PageMetadata, extract_page_metadata
from ..search.index_store import dump_index, load_previous_doc_ids
from ..search.ranking import compute_pagerank_boosts
from ..storage import content_available, init_db, iter_page_metas, try_read_content, url_hash
from ..storage.embedding_cache import load_embedding, save_embedding
from ..storage.models import Link, Page
from ..storage.page import normalize_url
from ..storage.token_cache import load_tokens, save_tokens
from .embeddings import (
    DocumentEmbedding,
    EmbeddingIndex,
    build_document_text,
    build_embedding_index,
    chunk_text,
    chunks_for_content,
    embed_text_chunks,
)
from .embedding_batch import take_embed_batch
from .embedding_config import (
    EMBED_CHUNK_BUDGET,
    EMBED_SOLO_DOC_CHUNKS,
    EMBED_WAIT_TIMEOUT_SEC,
    EXTRACT_PAGE_BATCH,
    EXTRACT_POOL_RECYCLE,
    EXTRACT_WORKERS,
    MAX_CHUNKS_PER_DOC,
)
from .embedding_model import is_model_installed, load_embedder
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


def filter_pages_with_content(pages: list[dict]) -> tuple[list[dict], int]:
    valid: list[dict] = []
    missing = 0
    for page in pages:
        if content_available(page["contentHash"]):
            valid.append(page)
        else:
            missing += 1
    return valid, missing


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
    html = try_read_content(content_hash)
    if html is None:
        return None
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


def _extract_document_chunks(meta: dict) -> tuple[str, str, list[str]] | None:
    url = meta["url"]
    content_hash = meta["contentHash"]
    doc_id = meta["urlHash"]
    html = try_read_content(content_hash)
    if html is None:
        return None
    if not is_indexable_page(url, html):
        return None
    page_meta = extract_page_metadata(html, url)
    chunks = chunk_text(build_document_text(page_meta))
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        chunks = chunks[:MAX_CHUNKS_PER_DOC]
    if not chunks:
        return None
    return doc_id, content_hash, chunks


def _embed_extracted_documents(
    extracted: list[tuple[str, str, list[str]]],
    *,
    embedder: TextEmbedding,
    progress,
    task,
) -> tuple[dict[str, DocumentEmbedding], int]:
    doc_embeddings: dict[str, DocumentEmbedding] = {}
    embedded = 0

    all_chunks: list[str] = []
    spans: list[tuple[str, str, list[str], int, int]] = []

    for doc_id, content_hash, chunks in extracted:
        start = len(all_chunks)
        all_chunks.extend(chunks)
        spans.append((doc_id, content_hash, chunks, start, len(all_chunks)))

    vectors = run_with_progress_refresh(
        progress,
        embed_text_chunks,
        all_chunks,
        embedder,
    )
    for doc_id, content_hash, chunks, start, end in spans:
        doc_vectors = vectors[start:end]
        save_embedding(content_hash, doc_vectors)
        doc_embeddings[doc_id] = DocumentEmbedding(chunks=chunks, vectors=doc_vectors)
        embedded += 1
        progress.advance(task, 1)

    return doc_embeddings, embedded


def _flush_embed_batch(
    ready_to_embed: list[tuple[str, str, list[str]]],
    *,
    embedder: TextEmbedding,
    doc_embeddings: dict[str, DocumentEmbedding],
    progress,
    task,
) -> int:
    batch = take_embed_batch(
        ready_to_embed,
        chunk_budget=EMBED_CHUNK_BUDGET,
        solo_doc_chunks=EMBED_SOLO_DOC_CHUNKS,
    )
    if not batch:
        return 0

    batch_embeddings, batch_embedded = _embed_extracted_documents(
        batch,
        embedder=embedder,
        progress=progress,
        task=task,
    )
    doc_embeddings.update(batch_embeddings)
    return batch_embedded


def _extract_and_embed_pending(
    pending: list[dict],
    *,
    embedder: TextEmbedding,
    progress,
    task,
) -> tuple[dict[str, DocumentEmbedding], int, int]:
    doc_embeddings: dict[str, DocumentEmbedding] = {}
    embedded = 0
    skipped_extract = 0
    ready_to_embed: list[tuple[str, str, list[str]]] = []

    with ProcessPoolExecutor(
        max_workers=EXTRACT_WORKERS,
        max_tasks_per_child=EXTRACT_POOL_RECYCLE,
    ) as pool:
        pending_iter = iter(pending)
        in_flight: dict = {}

        def submit_next() -> None:
            meta = next(pending_iter, None)
            if meta is not None:
                in_flight[pool.submit(_extract_document_chunks, meta)] = meta

        for _ in range(min(EXTRACT_PAGE_BATCH, len(pending))):
            submit_next()

        while in_flight or ready_to_embed:
            done: set = set()
            if in_flight:
                done, _ = wait(
                    in_flight,
                    timeout=EMBED_WAIT_TIMEOUT_SEC if ready_to_embed else None,
                    return_when=FIRST_COMPLETED,
                )

            for future in done:
                del in_flight[future]
                result = future.result()
                if result is None:
                    skipped_extract += 1
                    progress.advance(task, 1)
                else:
                    ready_to_embed.append(result)
                submit_next()

            while ready_to_embed:
                embedded += _flush_embed_batch(
                    ready_to_embed,
                    embedder=embedder,
                    doc_embeddings=doc_embeddings,
                    progress=progress,
                    task=task,
                )
                if in_flight:
                    break

    return doc_embeddings, embedded, skipped_extract


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


def _load_cached_embeddings(
    doc_ids: list[str],
    *,
    force: bool,
) -> tuple[dict[str, DocumentEmbedding], list[dict], int]:
    doc_embeddings: dict[str, DocumentEmbedding] = {}
    pending: list[dict] = []
    cached = 0

    for doc_id in doc_ids:
        page = Page.get_by_id(doc_id)
        if not force:
            vectors = load_embedding(page.content_hash)
            if vectors is not None:
                chunks = chunks_for_content(page.content_hash, page.url)
                if chunks is not None and len(chunks) == len(vectors):
                    doc_embeddings[doc_id] = DocumentEmbedding(chunks=chunks, vectors=vectors)
                    cached += 1
                    continue
        pending.append(
            {"urlHash": doc_id, "url": page.url, "contentHash": page.content_hash}
        )

    return doc_embeddings, pending, cached


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
    all_pages = list(iter_page_metas())
    pages, missing_content = filter_pages_with_content(all_pages)
    domains = Counter(page["url"].split("/")[2] for page in pages)
    plan = plan_index(pages, force=args.force)
    previous_doc_ids = load_previous_doc_ids()

    console = Console()
    console.print(
        f"Found [bold]{len(pages)}[/bold] webpages"
        f" from [bold]{len(domains)}[/bold] unique domains"
    )
    if missing_content:
        console.print(
            f"[yellow]Skipping {missing_content} pages with missing content files[/yellow]"
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
    console.print("[dim]Building BM25 index...[/dim]")
    bm25 = BM25Okapi(corpus_tokens)
    console.print("[dim]Computing PageRank...[/dim]")
    pagerank = _build_pagerank(doc_ids)
    embedding_index = None
    if is_model_installed():
        console.print("[dim]Loading embedding model...[/dim]")
        embedder = load_embedder()

        doc_embeddings, pending, cached = _load_cached_embeddings(
            doc_ids,
            force=args.force,
        )

        embedded = 0
        skipped_extract = 0
        if pending:
            console.print(
                f"[dim]Extracting and embedding {len(pending)} pages"
                f" ({EXTRACT_WORKERS} workers, up to {EXTRACT_PAGE_BATCH} in flight)...[/dim]"
            )
            embedding_progress = make_determinate_progress()
            with embedding_progress:
                embedding_task = embedding_progress.add_task(
                    "Building semantic embeddings",
                    total=len(doc_ids),
                )
                embedding_progress.advance(embedding_task, cached)
                new_embeddings, embedded, skipped_extract = _extract_and_embed_pending(
                    pending,
                    embedder=embedder,
                    progress=embedding_progress,
                    task=embedding_task,
                )
                doc_embeddings.update(new_embeddings)
        elif cached:
            console.print("[dim]All embeddings loaded from cache[/dim]")

        console.print(
            f"[dim]{cached} embedding cache hits, {embedded} embedded[/dim]"
        )
        embedding_index = build_embedding_index(doc_ids, doc_embeddings)
    else:
        console.print(
            "[dim]Semantic embeddings skipped — run `uv run setup-models` to enable them[/dim]"
        )
    console.print("[dim]Writing index files...[/dim]")
    dump_index(bm25, doc_ids, pagerank, embedding_index)
    console.print(
        f"[green]Built index with {len(doc_ids)} documents"
        f", PageRank boosts"
        f"{', and semantic embeddings' if embedding_index else ''}[/green]"
    )


if __name__ == "__main__":
    main()
