import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

_INDEX_FILE = Path("data") / "index.pkl"
_DOCS_FILE = Path("data") / "docs.json"


def dump_index(bm25: BM25Okapi, doc_ids: list[str]) -> None:
    _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)


def load_index() -> tuple[BM25Okapi, list[str]]:
    with open(_INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["doc_ids"]


def dump_docs(docs: dict[str, dict[str, str]]) -> None:
    _DOCS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)


def load_docs() -> dict[str, dict[str, str]]:
    with open(_DOCS_FILE, encoding="utf-8") as f:
        return json.load(f)
