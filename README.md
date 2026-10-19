# MAIO - Virtual Diabetes Clinic Triage
Case assignment VT2026 Stockholm University, for course MAIO Group G.

## What it is

A hospital runs a virtual diabetes clinic where nurses review hundreds of patient check-ins each week to decide who needs a follow-up call. This service predicts short-term disease progression as a continuous risk score, so the triage dashboard can sort patients by score (highest first) and call them in that order.

- `POST /predict` takes the 10 patient features and returns `{"prediction": <float>}`. Higher means more expected progression.
- `GET /health` returns `{"status": "ok", "model_version": "..."}` for monitoring.

The model is trained on the open scikit-learn diabetes dataset (`load_diabetes`) as a stand-in for de-identified EHR data. The target is used as the progression index. Training, testing, the Docker build and releases all run in GitHub Actions.

| Version | Model                                 | Change                                        |
|---------|---------------------------------------|-----------------------------------------------|
| v0.1    | `StandardScaler` + `LinearRegression` | Baseline API and Docker image                 |
| v0.2    | Same model                            | Calibrated high-risk flag cutoff (in metrics), smaller image |
> For v0.2, Ridge, RidgeCV, LassoCV, RandomForest and GradientBoosting were compared with the baseline by 5-fold cross-validated RMSE on the training split. None was meaningfully better (the best, Ridge, was 0.02 lower against a spread of about 2.4 between folds), so the baseline model was kept.

What changed and why, with metrics for both versions, is in [CHANGELOG.md](CHANGELOG.md).

## How to run it
The images are public on GHCR, so no login is needed:

```bash
docker pull ghcr.io/tictactoers/maio:v0.1
docker run -it -p 80:80 ghcr.io/tictactoers/maio:v0.1
```

Same for v0.2, just change the tag. Both use port 80, so stop the v0.1 container first.

Check that it's up:

```bash
curl http://localhost:80/health
# {"status":"ok","model_version":"v0.1"}   (v0.2 reports "v0.2")
```

Get a prediction (sample payload):

```bash
curl -X POST http://localhost:80/predict -H "Content-Type: application/json" -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
# {"prediction":235.94963722176266}
```

That's the v0.1 value (the last digits can differ slightly between machines). v0.2 returns the same number, because only the high-risk flag changed (see CHANGELOG.md).

On Windows, use PowerShell and write `curl.exe` instead of `curl`. Also change the quotes inside JSON:

```powershell
curl.exe http://localhost:80/health

'{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}' | curl.exe -X POST http://localhost:80/predict -H "Content-Type: application/json" -d "@-"
```

All 10 features are required floats on the scaled range of the sklearn diabetes dataset. Each feature column is scaled to unit norm (sum of squares = 1), so every value in the dataset lies within -1 to 1, and in practice within about -0.2 to 0.2. We reject anything outside -1.0 to 1.0 with a 422, mainly so raw values like `"age": 45` aren't accepted by accident. Extra fields are ignored.

## Errors
Bad input gives HTTP 422 with a JSON body. For example, the request above without `s6` returns:

```json
{
  "error": "Invalid input",
  "details": [
    {"field": "s6", "message": "Field required", "type": "missing"}
  ]
}
```

- `error` is always `"Invalid input"` for these.
- `details` has one entry per problem. `field` is the input that failed (or `body` if the JSON itself is broken), `message` says what is wrong, and `type` is the short code (`missing`, `float_parsing`, `less_than_equal`, `json_invalid`, ...).

If something breaks on our side you get a 500 with `{"error": "Internal error"}`. The details end up in the container logs.

## Running it locally
You need Python 3.13 (CI and the Docker image use 3.13.7).
On Windows, the `py` launcher picks the right version if you have several installed.

```bash
git clone https://github.com/TicTacToers/MAIO.git
cd MAIO
```

Create and activate a virtual environment. On Mac and Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first. It only applies to the current window. From here the commands are the same everywhere:

```bash
pip install -r requirements-dev.txt

ruff check .
ruff format --check .
pytest -q
python -m ml.train --out-dir artifacts
uvicorn app.main:app --port 8000
```

`ml.train` writes `artifacts/model.joblib` and `artifacts/metrics.json`. Both versions use the same model, so you should get a test RMSE of 53.8534 and a 5-fold CV RMSE of 55.3946 ± 2.3615. They differ in the high-risk flag: v0.1 flags at 214.0 (precision 1.0, recall 0.2857), v0.2 at the calibrated cutoff 197.0 (precision 0.8571, recall 0.5714). To get the exact code of a release, run `git checkout v0.1` (or `v0.2`) first. The numbers for each version are in CHANGELOG.md.

The experiments behind the v0.2 changelog can be re-run with `python -m experiments.compare_models` (candidate models by CV RMSE) and `python -m experiments.calibrate_flag` (flag cutoffs). They are not part of the Docker image.

If MLflow is installed (it is in requirements-dev.txt), the training run is logged there as well. `mlflow ui` shows it under "diabetes-progression". Use `--no-mlflow` to skip it.

Why the metrics are the same on every machine (even if the last digits of the prediction can differ between OS): seed and split are fixed in `ml/config.py`, all dependencies are pinned, the Docker base image is pinned by digest, one test trains twice and checks that the results match, and CI checks that the model inside the image is the same one CI trained.

To build the image yourself:

```bash
docker build -t diabetes-triage:local .
docker run -it -p 80:80 diabetes-triage:local
```

The model is trained during the build and baked into the image. The container runs as a non-root user and has a healthcheck.

## CI and releases
- `ci.yml` runs on every push to main and on every PR: lint, tests, a training run (model and metrics are uploaded as an artifact), the Docker build and a smoke test against the running container.
- `release.yml` runs when a tag like `v0.1` is pushed. It checks that the tag matches `MODEL_VERSION` and that CHANGELOG.md has a section for it, builds and tests the image, pushes it to GHCR and creates a GitHub Release with `metrics.json` attached.
- `release.yml` can also be started by hand from the Actions tab ("Run workflow"). That is a dry run: the same checks, build and smoke test, but nothing is pushed or released.

To release a new version, bump `MODEL_VERSION` in `ml/config.py` and add the CHANGELOG section in a PR. After the merge, tag main (`git tag v0.2`) and push the tag (`git push origin v0.2`). A newly published package starts out private, so switch it to public in the package settings. Published tags are never moved.

## Measuring image size and speed

The script `scripts/measure.sh` measures two things:

- how big the Docker image is (in MB)
- how fast `/predict` answers (in milliseconds)

We use these numbers to compare v0.1 and v0.2 in `CHANGELOG.md`.

```bash
# Measure the current code
bash scripts/measure.sh

# Measure a published version
bash scripts/measure.sh ghcr.io/tictactoers/maio:v0.1
```

The script starts the container, sends 20 practice requests, and then times 200 identical requests. It prints the typical response time (median) and the slow end (95th percentile). At the end it prints two lines you can copy into the CHANGELOG table.

It uses port 8000, so it can run while another container uses port 80.
Results depend on the machine. Keep in mind for comparability.
You need Docker, `curl` and `bash` (Mac, Linux).
On Windows, run the script from Git Bash with Docker Desktop running.

## AI disclosure

We used AI assistants (Claude) during this project, mainly to help format the documentation (README.md, CHANGELOG.md) and to write tests. All AI-assisted changes were reviewed by the team and went through the same PR review and CI checks as the rest of the code.