from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import FEATURES, MODEL_VERSION, RANDOM_STATE, TEST_SIZE


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = load_diabetes()
    if list(data.feature_names) != FEATURES:
        raise RuntimeError(f"Unexpected feature order: {data.feature_names}")
    return data.data, data.target


def train(out_dir: Path) -> dict:
    np.random.seed(RANDOM_STATE)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    preds = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    metrics = {
        "model_version": MODEL_VERSION,
        "model": type(pipeline.named_steps["model"]).__name__,
        "rmse": round(rmse, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_seconds": round(train_seconds, 4),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "model_version": MODEL_VERSION, "features": FEATURES},
        out_dir / "model.joblib",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def log_to_mlflow(metrics: dict, out_dir: Path) -> bool:
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; skipping experiment tracking")
        return False

    mlflow.set_experiment("diabetes-progression")
    with mlflow.start_run(run_name=metrics["model_version"]):
        mlflow.log_params(
            {k: metrics[k] for k in ("model_version", "model", "test_size", "random_state")}
        )
        mlflow.log_metrics({"rmse": metrics["rmse"], "train_seconds": metrics["train_seconds"]})
        mlflow.log_artifact(str(out_dir / "metrics.json"))
        mlflow.log_artifact(str(out_dir / "model.joblib"))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    args = parser.parse_args()
    metrics = train(args.out_dir)
    if not args.no_mlflow:
        log_to_mlflow(metrics, args.out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
