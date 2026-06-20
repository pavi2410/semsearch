import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

INDEX_DIR = Path("data") / "index"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
PAGERANK_FILE = INDEX_DIR / "pagerank.json"
BM25_FILE = INDEX_DIR / "bm25.pkl"
INDEX_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump_index(
    bm25: BM25Okapi,
    doc_ids: list[str],
    pagerank: dict[str, float],
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": INDEX_VERSION,
        "built_at": _now(),
        "doc_count": len(doc_ids),
        "doc_ids": doc_ids,
    }

    manifest_tmp = INDEX_DIR / "manifest.json.tmp"
    pagerank_tmp = INDEX_DIR / "pagerank.json.tmp"
    bm25_tmp = INDEX_DIR / "bm25.pkl.tmp"

    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(pagerank_tmp, "w", encoding="utf-8") as f:
        json.dump(pagerank, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(bm25_tmp, "wb") as f:
        pickle.dump(bm25, f)

    os.replace(manifest_tmp, MANIFEST_FILE)
    os.replace(pagerank_tmp, PAGERANK_FILE)
    os.replace(bm25_tmp, BM25_FILE)


def load_index() -> tuple[BM25Okapi, list[str], dict[str, float]]:
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(PAGERANK_FILE, encoding="utf-8") as f:
        pagerank = json.load(f)
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)
    return bm25, manifest["doc_ids"], pagerank


def load_previous_doc_ids() -> list[str]:
    if not MANIFEST_FILE.exists():
        return []
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest.get("doc_ids", [])
