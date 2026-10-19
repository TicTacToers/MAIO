# Changelog

Test metrics are from the held-out test split (20%, `random_state=42`, n_test = 89).
CV RMSE is 5-fold cross-validation on the training split only.


## [v0.1] — Baseline

- Model: `StandardScaler` + `LinearRegression`
- API: `GET /health`, `POST /predict`, with JSON 422 errors on invalid input (features must be within ±1.0)
- Docker: multi-stage build, model baked in, non-root user, port 80, healthcheck
- High-risk flagging (reported only in metrics.json): a patient is flagged when the predicted progression score is at or above the 75th percentile of the real progression values for training patients

| Metric                       | v0.1             |
|------------------------------|------------------|
| RMSE (test)                  | 53.8534          |
| CV RMSE (5-fold, mean ± std) | 55.3946 ± 2.3615 |
| High-risk precision          | 1.0              |
| High-risk recall             | 0.2857           |


## [v0.2] — Calibrated high-risk flag, smaller image

### What changed
- Model and preprocessing are unchanged (`StandardScaler` + `LinearRegression`). Five alternatives were compared by 5-fold cross-validated RMSE on the training split and none clearly beat the baseline: the best one (Ridge) is 0.02 lower, far below the spread between folds of about 2.4 (table below).
- The high-risk flag now uses a calibrated cutoff: the cutoff with the best F1 on out-of-fold predictions of the training split, searched from the lowest prediction up to the v0.1 cutoff in steps of 1. It is 197.0, instead of the 214.0 that also defines "truly high risk". The test split is used once, for reporting.
- `metrics.json` has new fields: `high_risk_f1`, `calibrated_cutoff`, `calibrated_precision`, `calibrated_recall`, `calibrated_f1`.
- `/predict` is unchanged. The flag is reported in the metrics only.
- Docker: the runtime image no longer contains the bundled test suites of numpy, scipy and scikit-learn, or `pip`. None of them are used at runtime. The model and all outputs are unchanged.
- Reproduce: `python -m experiments.compare_models` and `python -m experiments.calibrate_flag` (pinned environment).

### Metrics (side by side)

| Metric                       | v0.1             | v0.2             | Δ        |
|------------------------------|------------------|------------------|----------|
| RMSE (test)                  | 53.8534          | 53.8534          | 0        |
| CV RMSE (5-fold, mean ± std) | 55.3946 ± 2.3615 | 55.3946 ± 2.3615 | 0        |
| Flag cutoff                  | 214.0            | 197.0            | -17.0    |
| Precision (test)             | 1.0              | 0.8571           | -0.1429  |
| Recall (test)                | 0.2857           | 0.5714           | +0.2857  |
| F1 (test)                    | 0.4444           | 0.6857           | +0.2413  |
| F1 (training, out-of-fold)   | 0.6389           | 0.6941           | +0.0552  |
| Image size (unpacked)        | 480.9 MB         | 403.4 MB         | -77.5 MB |
| p50 /predict latency         | 2.15 ms          | 2.10 ms          | -0.05 ms |

Image size and latency were measured with `scripts/measure.sh` (`N=1000`), with v0.1 (from its tag) and v0.2 both built locally on the same machine (MacBook Air M4). Image size is the unpacked size, the sum of the image layers. Latency depends on the machine and varies by about 0.5 ms between runs, so the two versions are equally fast.

### Candidate models (CV RMSE on the training split, mean ± std)

| Model                   | CV RMSE          |
|-------------------------|------------------|
| LinearRegression (v0.1) | 55.3946 ± 2.3615 |
| Ridge alpha=1           | 55.3742 ± 2.3732 |
| RidgeCV                 | 55.3992 ± 2.2690 |
| LassoCV                 | 55.5769 ± 2.2916 |
| RandomForest            | 58.9308 ± 1.9312 |
| GradientBoosting        | 61.1980 ± 2.8104 |

### Why this is an improvement
- v0.1 flags only patients with a predicted score of 214 or more. Linear predictions are pulled toward the average, so it misses most high-risk patients (recall 0.2857).
- v0.2 picks the cutoff on training data only. Recall doubles to 0.5714 and F1 rises from 0.4444 to 0.6857, at the cost of precision (1.0 to 0.8571).
- The training split points the same way with a smaller gain (F1 0.6389 at cutoff 214, 0.6941 at 197). That value is optimistic, because 197 was picked as the best of many cutoffs on the same predictions. The test split has only 89 rows, so a few patients move its numbers a lot.
- Whether a lower cutoff is right depends on the clinic: catching more patients means more follow-up calls.
- The image is 77.5 MB (16%) smaller, so it pulls faster, with the same model, predictions and latency.
