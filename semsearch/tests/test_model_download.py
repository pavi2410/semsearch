from pathlib import Path
from unittest.mock import MagicMock

from semsearch.index.embedding_model import HF_MODEL_REPO
from semsearch.model_download import clean_broken_model_cache, download_embedding_model


def test_clean_broken_model_cache_removes_incomplete_local_install(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("semsearch.model_download.LOCAL_MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr("semsearch.index.embedding_model.LOCAL_MODEL_DIR", tmp_path / "model")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    clean_broken_model_cache(model_dir)

    assert not model_dir.exists()


def test_clean_broken_model_cache_keeps_complete_snapshots(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.onnx").write_text("onnx", encoding="utf-8")

    clean_broken_model_cache(model_dir)

    assert model_dir.exists()


def test_download_embedding_model_uses_huggingface_hub(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "all-MiniLM-L6-v2-onnx"
    monkeypatch.setattr("semsearch.model_download.LOCAL_MODEL_DIR", model_dir)
    monkeypatch.setattr("semsearch.index.embedding_model.LOCAL_MODEL_DIR", model_dir)

    def fake_download(**kwargs):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.onnx").write_text("onnx", encoding="utf-8")
        return str(model_dir)

    snapshot_download = MagicMock(side_effect=fake_download)
    monkeypatch.setattr(
        "semsearch.model_download.snapshot_download",
        snapshot_download,
    )

    result = download_embedding_model()

    assert result == model_dir
    snapshot_download.assert_called_once()
    assert snapshot_download.call_args.kwargs["repo_id"] == HF_MODEL_REPO
    assert snapshot_download.call_args.kwargs["local_dir"] == str(model_dir)
