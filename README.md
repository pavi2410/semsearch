# Semsearch

This project implements a web search engine command-line interface (CLI) using the BM25 (Best Matching 25) algorithm. It is written in Python and uses `uv` for project management.

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Installation

```sh
uv sync
```

### Usage

First, crawl websites and index their content:

```sh
uv run crawl
uv run index
```

Then, search the index:

```sh
uv run search "your query"
```

## How It Works

1. **Crawling** — DFS-based web crawler collects HTML content from starting URLs, extracts links for further crawling, and caches pages in `data/webpages/`
2. **Indexing** — Parses HTML (title + body text via BeautifulSoup), runs an NLP pipeline (tokenize, stem, stopwords, negation propagation), then builds a BM25 index stored in `data/`
3. **Searching** — Preprocesses the query through the same NLP pipeline, scores documents with BM25, and displays top-10 results with fuzzy-matched highlights

## Project Structure

```
semsearch/
├── cli.py            # search CLI entry point
├── crawler.py        # web crawler entry point
├── indexer.py        # BM25 indexer entry point
└── core/
    ├── config.py     # path configuration
    ├── html_utils.py # HTML title/text/link extraction (BeautifulSoup)
    ├── nlp.py        # NLP pipeline (tokenize, stem, stopwords, negations)
    └── search.py     # BM25 search function
```

## License

MIT — see [LICENSE](LICENSE)
