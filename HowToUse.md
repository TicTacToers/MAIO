# Diabetes Progression Triage — ML Service

> AI Generated file, to help you get started. Only for demonstration purposes.

A small FastAPI service that predicts short-term diabetes **progression** (a continuous
risk score, higher = worse) so a virtual clinic can sort patients for follow-up.
It is trained on scikit-learn's diabetes regression dataset and shipped as a
self-contained Docker image on GHCR.

| Endpoint        | Purpose                                   |
|-----------------|-------------------------------------------|
| `GET /health`   | Liveness + which model version is loaded  |
| `POST /predict` | Progression score for one patient         |
| `GET /docs`     | Interactive OpenAPI docs (FastAPI)        |

## Run the released image

```bash
docker pull ghcr.io/<namespace>/<repo>:v0.1
docker run -it -p 80:80 ghcr.io/<namespace>/<repo>:v0.1
```

```bash
curl http://localhost:80/health
# {"status":"ok","model_version":"v0.1"}
```

## Sample request

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
       "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
# {"prediction": 235.95}
```

All 10 features are **required** floats, in the standardized scale of the
scikit-learn dataset (all dataset values lie within ±0.2). Values outside **[-0.5, 0.5]**
are rejected, so raw clinical values (e.g. `"age": 45`) get a 422 instead of a
nonsense score. Extra fields are ignored. `NaN`/`Infinity` are rejected.

## Error responses

Invalid input (missing field, wrong type, malformed JSON) returns **HTTP 422**:

```json
{
  "error": "Invalid input",
  "details": [
    {"field": "s6", "message": "Field required", "type": "missing"}
  ]
}
```

| Key               | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `error`           | Always `"Invalid input"` for validation failures               |
| `details[].field` | The offending field name (`"body"` if the whole body is bad)   |
| `details[].message` | Human-readable reason                                        |
| `details[].type`  | Machine-readable reason (`missing`, `float_parsing`, `json_invalid`, …) |

## Develop locally

Requires Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

ruff check . && ruff format --check .   # lint
pytest -q                                # tests (trains a model in a temp dir)

python -m ml.train --out-dir artifacts   # train -> artifacts/model.joblib + metrics.json
uvicorn app.main:app --port 8000         # serve locally on :8000
```

### Experiment tracking (MLflow)

`ml.train` logs params, RMSE and the model to MLflow when it is installed
(it is in `requirements-dev.txt`, deliberately **not** in the Docker image, to keep it slim).

```bash
python -m ml.train --out-dir artifacts
mlflow ui            # open http://127.0.0.1:5000, experiment "diabetes-progression"
```

Use `--no-mlflow` to skip tracking.

## Build the image yourself

```bash
docker build -t diabetes-triage:local .
docker run -it -p 80:80 diabetes-triage:local
```

The Dockerfile is multi-stage. The builder stage installs pinned dependencies and
**trains the model during the build**, and the runtime stage copies only the venv,
the model and `app/`. The image runs as a non-root user and has a `HEALTHCHECK`.

## Reproducibility

- Fixed seed and split: `RANDOM_STATE=42`, `TEST_SIZE=0.2` in `ml/config.py`.
- Every dependency, including transitive ones, is pinned with `==` in `requirements.txt`.
- The base image is pinned (`python:3.13.7-slim`).
- The model is retrained from code in every build, so the same commit gives the same model and RMSE.
  `tests/test_train.py` asserts that two training runs produce identical metrics.

## Project layout

```
app/main.py        FastAPI app: /health, /predict, 422 error handler
ml/config.py       MODEL_VERSION, seed, split, feature order
ml/train.py        training + metrics + optional MLflow logging
tests/             API + training tests
Dockerfile         multi-stage: train -> slim runtime on port 80
.github/workflows  CI (lint, test, train, artifacts, docker smoke)
```

## Releasing

1. Bump `MODEL_VERSION` in `ml/config.py` and update `CHANGELOG.md` in a PR.
2. After merge, tag the merge commit: `git tag v0.X && git push origin v0.X`.
3. The release workflow builds and smoke-tests the image, pushes it to GHCR, and creates the GitHub Release.

Never move or re-push an existing tag. `v0.1` must keep serving the v0.1 model.
