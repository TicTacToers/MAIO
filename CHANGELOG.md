# Changelog

Test metrics are from the held-out test split (20%, `random_state=42`, n_test = 89).
CV RMSE is 5-fold cross-validation on the training split only.

## [v0.1] — Baseline

- Model: `StandardScaler` + `LinearRegression`
- API: `GET /health`, `POST /predict`, with JSON 422 errors on invalid input
- Docker: multi-stage build, model baked in, non-root user, port 80, healthcheck
- High-risk flagging (only reported metrics.json): a patient is flagged when the predicted progression score is above the 75th percentile of the real progression values for training patients.
| Metric                       | v0.1             |
|------------------------------|------------------|
| RMSE (test)                  | 53.8534          |
| CV RMSE (5-fold, mean ± std) | 55.3946 ± 2.3615 |
| High-risk precision          | 1.0              |
| High-risk recall             | 0.2857           |

<!--
## [v0.2] — <what improved>

### What changed
- ...

### Metrics (side by side)
| Metric               | v0.1    | v0.2 | Δ |
|----------------------|---------|------|---|
| RMSE                 | 53.8534 |      |   |
| Precision @ thresh   | 1.0     |      |   |
| Recall @ thresh      | 0.2857  |      |   |
| Image size           |         |      |   |
| p50 /predict latency |         |      |   |

### Why this is an improvement
- ...
-->
