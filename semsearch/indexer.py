import hashlib
import json

from rank_bm25 import BM25Okapi

from .config import DOCS_FILE, INDEX_FILE, WEBPAGES_DIR
from .html_utils import extract_metadata
from .nlp import preprocess


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
    params = {
        "doc_ids": doc_ids,
        "corpus_tokens": corpus_tokens,
        "k1": bm25.k1,
        "b": bm25.b,
    }

    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)

    print(f"Wrote {DOCS_FILE} ({len(docs)} docs)")
    print(f"Wrote {INDEX_FILE}")


if __name__ == "__main__":
    main()
