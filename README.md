# Semsearch
This project implements a web search engine command-line interface (CLI) using the BM25 (Best Matching 25) algorithm. It is written in TypeScript and utilizes Bun APIs for improved performance.

![image](https://github.com/user-attachments/assets/c25dfcf4-b7ce-4c16-a0d2-8dad3785ba55)

## Getting Started

### Installation

The latest version of Bun is required.

Install dependencies:
```   
bun install
```

### Usage

First, crawl websites and index their content:
```
bun crawl
bun index
```

Then, use the CLI to search:
```
bun search [search terms ...]
```

## How It Works

1. **Crawling**: The engine crawls specified websites using Depth-first search and collects web pages' HTML content. It also extracts links to other pages for further crawling. This process outputs the content in the `webpages` directory.
2. **Indexing**: It processes the collected pages and builds an index using the BM25 algorithm. This process outputs a list of documents and their corresponding BM25 scores as `docs.json` and `index.json` files respectively.
3. **Searching**: Users can input search queries, and the engine returns top-10 relevant results ranked by their BM25 scores.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
