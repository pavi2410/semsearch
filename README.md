# Semsearch

A personal web search engine CLI: crawl pages, build a BM25 index, and search your corpus with lexical ranking plus metadata signals (PageRank, recency, HTTPS, title matching).

Written in Python and managed with [uv](https://docs.astral.sh/uv/).

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.14+

### Installation

```sh
uv sync
```

### One-time setup

Initialize the database and download the local embedding model used for semantic search:

```sh
uv run migrate
uv run setup-models
```

`setup-models` downloads `Qdrant/all-MiniLM-L6-v2-onnx` (~90MB) into `data/models/fastembed/`. This is a one-time step unless you delete that folder or pass `--force`.

Optional but recommended for faster, more reliable downloads:

```sh
export HF_TOKEN=hf_...   # https://huggingface.co/settings/tokens
uv run setup-models
```

Indexing automatically builds semantic embeddings when the model is installed. Without it, `uv run index` still builds the BM25 and PageRank index.

### Usage

Crawl starting URLs, index the corpus, then search:

```sh
uv run crawl
uv run index
uv run search "your query"
```

Re-index only changed pages (default). Force a full rebuild:

```sh
uv run index --force all
```

## How It Works

### 1. Crawling

DFS crawler with robots.txt and sitemap support. Fetched HTML is stored under `data/webpages/`; page metadata and the link graph live in SQLite (`data/semsearch.db`).

Each page records:

- Title, description, canonical URL
- Open Graph tags
- JSON-LD article types and publish/modified dates
- Main body text (prefers `<main>` / `<article>`, strips nav/footer chrome)
- Outbound links (for PageRank)
- Detected language

Non-HTML and JSON API responses are skipped at crawl and index time.

### 2. Indexing

Changed pages are tokenized through an NLP pipeline (tokenize → stopwords → Porter stem → negation propagation). Tokens are cached by content hash for incremental rebuilds.

The search index is split under `data/index/`:

| File | Contents |
|------|----------|
| `manifest.json` | Index version, build time, document IDs |
| `pagerank.json` | Precomputed PageRank boosts per document |
| `bm25.pkl` | BM25Okapi corpus |
| `embeddings.pkl` | Chunk embeddings for hybrid semantic search |

Semantic embeddings use the local ONNX model installed by `uv run setup-models`.

### 3. Searching

Queries use the same NLP preprocessing as the index. Final score combines:

| Signal | Notes |
|--------|-------|
| **BM25** | Bag-of-words relevance over title + description + body |
| **Semantic** | Local MiniLM embeddings fused with BM25 via reciprocal rank fusion |
| **Title match** | Extra boost when the query phrase or terms appear in the title |
| **PageRank** | Link-graph popularity (1.0–1.3×, log-scaled) |
| **Recency** | Fresher pages score higher via publish/modified/fetched timestamps |
| **HTTPS** | Small preference for `https://` over `http://` |
| **Dampening** | Metadata boosts are reduced for documents that already rank highly on BM25 |

Results show title, URL, snippet, score, and language tag (e.g. `[en]`).

## Project Structure

```
semsearch/
├── cli.py                 # search CLI
├── crawl/
│   ├── crawler.py         # web crawler
│   ├── metadata.py        # meta/OG/JSON-LD/link extraction
│   ├── main_content.py    # main-body text extraction
│   ├── language.py        # language detection
│   └── content_filter.py  # skip non-indexable content
├── index/
│   ├── indexer.py         # BM25 + PageRank index builder
│   ├── embedding_model.py # local ONNX model load
│   └── nlp.py             # tokenization pipeline
├── model_download.py      # Hugging Face model download for setup-models
├── search/
│   ├── search.py          # query + scoring
│   ├── ranking.py         # PageRank, recency, HTTPS, title boosts
│   ├── index_store.py     # index load/save
│   └── snippet.py         # result snippets
└── storage/
    ├── models.py          # SQLite schema (pages, links, token cache)
    └── content.py         # webpage file storage
```

## Roadmap

Tracked in GitHub issues. Up next:

- **#8** — Semantic / hybrid search (embeddings)
- **#7** — MCP server
- **#6** — llms.txt and ARD support
- **#3** — Remaining ranking signals (geo, performance, language filter)

## License

MIT — see [LICENSE](LICENSE)
