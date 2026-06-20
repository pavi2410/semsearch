from pathlib import Path

from fastembed import TextEmbedding

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_REPO = "Qdrant/all-MiniLM-L6-v2-onnx"
MODEL_CACHE_DIR = Path("data") / "models" / "fastembed"
LOCAL_MODEL_DIR = MODEL_CACHE_DIR / "all-MiniLM-L6-v2-onnx"

_embedder: TextEmbedding | None = None


def is_model_installed(model_dir: Path | None = None) -> bool:
    target = LOCAL_MODEL_DIR if model_dir is None else model_dir
    return (target / "config.json").exists() and (target / "model.onnx").exists()


def load_embedder(
    *,
    model_name: str = DEFAULT_MODEL,
    force_reload: bool = False,
) -> TextEmbedding:
    global _embedder

    if _embedder is not None and not force_reload:
        return _embedder

    _embedder = TextEmbedding(
        model_name=model_name,
        cache_dir=str(MODEL_CACHE_DIR),
        specific_model_path=str(LOCAL_MODEL_DIR),
    )
    return _embedder
