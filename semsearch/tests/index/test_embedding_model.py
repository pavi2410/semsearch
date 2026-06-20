from pathlib import Path

from semsearch.index.embedding_model import is_model_installed


def test_is_model_installed_requires_config_and_onnx(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert not is_model_installed(model_dir)

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert not is_model_installed(model_dir)

    (model_dir / "model.onnx").write_text("onnx", encoding="utf-8")
    assert is_model_installed(model_dir)
