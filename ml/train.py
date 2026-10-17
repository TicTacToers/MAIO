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
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import (
    CUTOFF_STEP,
    CV_FOLDS,
    FEATURES,
    HIGH_RISK_PERCENTILE,
    MODEL_VERSION,
    RANDOM_STATE,
    TEST_SIZE,
)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def hyperparams(pipeline: Pipeline) -> dict:
    return {
        k: v if isinstance(v, bool | int | float | str | None) else repr(v)
        for k, v in pipeline.get_params().items()
        if "__" in k
    }


def cv_rmse(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return -cross_val_score(build_pipeline(), X, y, cv=cv, scoring="neg_root_mean_squared_error")


def flag_scores(
    preds: np.ndarray, actual_high: np.ndarray, cutoff: float
) -> tuple[float, float, float]:
    flag = preds >= cutoff
    return (
        float(precision_score(actual_high, flag, zero_division=0)),
        float(recall_score(actual_high, flag, zero_division=0)),
        float(f1_score(actual_high, flag, zero_division=0)),
    )


def calibrate_cutoff(X: np.ndarray, y: np.ndarray, truth_cut: float) -> float:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(build_pipeline(), X, y, cv=cv)  # All validation folds
    actual_high = y >= truth_cut
    grid = np.append(
        np.arange(np.floor(oof.min()), truth_cut, CUTOFF_STEP), truth_cut
    )  # Try all cutoffs from lowest prediction to the high-risk threshold
    return float(
        max(grid, key=lambda c: flag_scores(oof, actual_high, c)[2])
    )  # Get the cutoff with the best F1


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
    cv_scores = cv_rmse(X_train, y_train)

    pipeline = build_pipeline()
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    preds = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Computations for predictions considered high Risk
    threshold = float(np.percentile(y_train, HIGH_RISK_PERCENTILE))
    actual_high = y_test >= threshold
    precision, recall, f1 = flag_scores(preds, actual_high, threshold)
    cutoff = calibrate_cutoff(
        X_train, y_train, threshold
    )  # Cutoff after calibration is the one maximizing F1 on the training split
    cal_precision, cal_recall, cal_f1 = flag_scores(preds, actual_high, cutoff)

    metrics = {
        "model_version": MODEL_VERSION,
        "model": type(pipeline.named_steps["model"]).__name__,
        "rmse": round(rmse, 4),
        "high_risk_percentile": HIGH_RISK_PERCENTILE,
        "high_risk_threshold": round(threshold, 4),
        "high_risk_precision": round(precision, 4),
        "high_risk_recall": round(recall, 4),
        "high_risk_f1": round(f1, 4),
        "calibrated_cutoff": round(cutoff, 4),
        "calibrated_precision": round(cal_precision, 4),
        "calibrated_recall": round(cal_recall, 4),
        "calibrated_f1": round(cal_f1, 4),
        "cv_rmse_mean": round(float(cv_scores.mean()), 4),
        "cv_rmse_std": round(float(cv_scores.std()), 4),
        "cv_folds": CV_FOLDS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_seconds": round(train_seconds, 4),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "hyperparams": hyperparams(pipeline),
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
        run_params = ("model_version", "model", "test_size", "random_state", "cv_folds")
        mlflow.log_params({k: metrics[k] for k in run_params} | metrics["hyperparams"])
        mlflow.log_metrics(
            {
                k: metrics[k]
                for k in (
                    "rmse",
                    "cv_rmse_mean",
                    "cv_rmse_std",
                    "high_risk_precision",
                    "high_risk_recall",
                    "calibrated_precision",
                    "calibrated_recall",
                    "calibrated_f1",
                    "train_seconds",
                )
            }
        )
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
