# MAIO - Virtual Diabetes Clinic Triage
Case assignment VT2026 Stockholm University, for course MAIO Group G.

## Context. 
A hospital runs a virtual diabetes clinic. Each week, nurses review hundreds of patient check-ins (vitals, labs, lifestyle notes) to decide who needs a follow-up call. Reviews are manual and time-consuming.

## Goal. 
Build a small ML service that predicts short-term disease progression and returns a continuous risk score. The clinic will use the score to prioritize follow-ups. Everything should be built in a pipeline using GitHub Actions (https://docs.github.com/en/actions). More info on the ML service below in Users & Flow.

## Data. 
For the assignment, use the open scikit-learn Diabetes regression dataset as a stand-in for de-identified EHR features:

```python
from sklearn.datasets import load_diabetes

Xy = load_diabetes(as_frame=True)

X = Xy.frame.drop(columns=["target"])

y = Xy.frame["target"]  # acts as a "progression index" (higher = worse)
```
> Treat `y` as a progression index: higher values ≈ greater deterioration risk (e.g., rising HbA1c/complications risk). In production, a clinic would train on real outcomes; here, we map the same mechanics to a safe, open dataset.

## Users & flow.

- Triage Nurse opens a dashboard sorted by predicted progression (descending).

- ML Service hosts /predict that the dashboard calls for each patient and /health for internal monitoring.

- MLOps Team (you) owns training, packaging, testing, and releasing the service.

## Non-functional constraints.

Portability: Docker image must be self-contained.
Observability: Return JSON errors on bad input.

## Reproducibility: 
Same code + GitHub Actions should retrain and rebuild deterministically.

## Iteration plan.

### Iteration 1 (v0.1)
Baseline: simple StandardScaler + LinearRegression. Report RMSE on a held-out split. Ship a working API & Docker image.

### Iteration 2 (v0.2)
Improvement: try Ridge/RandomForestRegressor or better preprocessing (feature scaling/selection), plus calibration of the score if you convert to a “high-risk” flag (e.g., threshold on predicted progression). Show metric deltas (RMSE; if you add a flag, also precision/recall at a threshold) in CHANGELOG.md.

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
You need Docker, `curl` and `bash` (Mac or Linux).
