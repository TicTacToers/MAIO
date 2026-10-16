import joblib
import numpy as np
from sklearn.datasets import load_diabetes

from ml.train import train

# Wall-clock time is the only field allowed to differ between two runs.
RUN_DEPENDENT = {"train_seconds"}


def _stable(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in RUN_DEPENDENT}


def test_retraining_reproduces_metrics_and_model(artifacts, tmp_path):
    out, first = artifacts
    second = train(tmp_path)
    assert _stable(first) == _stable(second)

    X = load_diabetes().data
    before = joblib.load(out / "model.joblib")["pipeline"].predict(X)
    after = joblib.load(tmp_path / "model.joblib")["pipeline"].predict(X)
    np.testing.assert_array_equal(before, after)
