import json

from ml.config import CV_FOLDS, HIGH_RISK_PERCENTILE, MODEL_VERSION
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


def test_metrics_include_cv_and_hyperparams(artifacts):
    _, metrics = artifacts
    assert metrics["cv_folds"] == CV_FOLDS
    assert 0 < metrics["cv_rmse_mean"] < 70
    assert metrics["cv_rmse_std"] >= 0
    assert metrics["hyperparams"], "hyperparams should be logged for any model"
    assert all("__" in k for k in metrics["hyperparams"])


def test_training_is_deterministic(artifacts, tmp_path):
    _, first = artifacts
    second = train(tmp_path)
    assert first["rmse"] == second["rmse"]


def test_high_risk_metrics_valid(artifacts):
    _, metrics = artifacts
    assert metrics["high_risk_percentile"] == HIGH_RISK_PERCENTILE
    assert metrics["high_risk_threshold"] > 0
    assert 0 <= metrics["high_risk_precision"] <= 1
    assert 0 <= metrics["high_risk_recall"] <= 1
