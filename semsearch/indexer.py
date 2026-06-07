import hashlib
import json

from rank_bm25 import BM25Okapi

from .core.config import WEBPAGES_DIR
from .core.html_utils import extract_metadata
from .core.index_store import dump_docs, dump_index
from .core.nlp import preprocess


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def main() -> None:
    files = sorted(WEBPAGES_DIR.glob("*.json"))
    print(f"Found {len(files)} webpages")

    docs: dict[str, dict[str, str]] = {}
    corpus_tokens: list[list[str]] = []
    doc_ids: list[str] = []

    for i, fp in enumerate(files, 1):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        url: str = data["url"]
        html: str = data["content"]
        title, text = extract_metadata(html)

        doc_id = _url_hash(url)
        docs[doc_id] = {"url": url, "title": title}
        doc_ids.append(doc_id)

        tokens = preprocess(f"{title} {text}")
        corpus_tokens.append(tokens)

        print(f"Indexed page {i} of {len(files)}", end="\r")

    print()

    bm25 = BM25Okapi(corpus_tokens)

    dump_index(bm25, doc_ids)
    dump_docs(docs)


if __name__ == "__main__":
    main()
