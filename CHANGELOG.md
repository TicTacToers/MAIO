# Changelog

All metrics are from the held-out test split (20%, `random_state=42`, n_test = 89).

## [v0.1] — Baseline

- Model: `StandardScaler` + `LinearRegression`
- API: `GET /health`, `POST /predict`, with JSON 422 errors on invalid input
- Docker: multi-stage build, model baked in, non-root user, port 80, healthcheck

| Metric | v0.1    |
|--------|---------|
| RMSE   | 53.8534 |

<!--
## [v0.2] — <what improved>

### What changed
- ...

### Metrics (side by side)
| Metric               | v0.1    | v0.2 | Δ |
|----------------------|---------|------|---|
| RMSE                 | 53.8534 |      |   |
| Precision @ thresh   | —       |      |   |
| Recall @ thresh      | —       |      |   |
| Image size           |         |      |   |
| p50 /predict latency |         |      |   |

### Why this is an improvement
- ...
-->
