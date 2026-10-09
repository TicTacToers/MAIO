import json

from ml.config import MODEL_VERSION
from ml.train import train


def test_training_writes_artifacts(artifacts):
    out, metrics = artifacts
    assert (out / "model.joblib").exists()
    saved = json.loads((out / "metrics.json").read_text())
    assert saved == metrics
    assert saved["model_version"] == MODEL_VERSION


def test_rmse_is_sane(artifacts):
    _, metrics = artifacts
    assert 0 < metrics["rmse"] < 70


def test_training_is_deterministic(artifacts, tmp_path):
    _, first = artifacts
    second = train(tmp_path)
    assert first["rmse"] == second["rmse"]
