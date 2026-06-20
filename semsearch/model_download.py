import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from .index.embedding_model import HF_MODEL_REPO, LOCAL_MODEL_DIR, is_model_installed


def clean_broken_model_cache(model_dir: Path = LOCAL_MODEL_DIR) -> None:
    """Remove incomplete model directories left by interrupted downloads."""
    if model_dir.exists() and not is_model_installed(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)


def download_embedding_model(*, force: bool = False) -> Path:
    """Download the ONNX embedding model used for semantic search."""
    LOCAL_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if is_model_installed() and not force:
        return LOCAL_MODEL_DIR

    if force and LOCAL_MODEL_DIR.exists():
        shutil.rmtree(LOCAL_MODEL_DIR, ignore_errors=True)

    clean_broken_model_cache(LOCAL_MODEL_DIR)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    snapshot_download(
        repo_id=HF_MODEL_REPO,
        local_dir=str(LOCAL_MODEL_DIR),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )

    if not is_model_installed():
        clean_broken_model_cache(LOCAL_MODEL_DIR)
        raise RuntimeError(
            f"Downloaded model at {LOCAL_MODEL_DIR} is incomplete. "
            "Check your network connection and retry `uv run setup-models`."
        )

    return LOCAL_MODEL_DIR
