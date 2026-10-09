# Team roadmap

The ground rule: **no direct pushes to `main`.** Every change goes on a branch, then a PR
that needs one teammate's approval and green CI, then a merge. Who did what is graded
through this history, so spread PRs across people.

## Done in the scaffold PR (`feat/scaffold`)
- FastAPI service with `/health`, `/predict` and JSON 422 errors
- v0.1 training (StandardScaler + LinearRegression), metrics.json, optional MLflow logging
- Multi-stage Dockerfile on port 80 with the model baked in, non-root user and a healthcheck
- CI on push/PR: lint, tests, training smoke, artifact upload, Docker build + container smoke
- README, CHANGELOG (v0.1)

## Step 0: Repo settings (repo owner, 10 min, before anything else)
1. Make the repo **public**.
2. Settings → Branches → add a ruleset for `main` that requires a PR, 1 approval, and the `CI` status checks. Block force pushes.
3. Add the teammates as collaborators.
4. Settings → Actions → General → Workflow permissions: **Read and write**, so the release workflow can push to GHCR and create Releases.

## Step 1: Release workflow (`feat/release-workflow`, owner: ___)
Create `.github/workflows/release.yml`, triggered on `push: tags: ['v*']`. It needs `permissions: contents: write, packages: write` and these steps:
1. Check that the tag matches `MODEL_VERSION` in `ml/config.py`, and fail if it doesn't.
2. Install deps, run tests, run `python -m ml.train --out-dir artifacts --no-mlflow` to get `metrics.json` for the release notes.
3. `docker build`, `docker run -d -p 80:80`, then curl `/health` (and check that `model_version == tag`), `/predict` and the 422 case. The `docker-smoke` job in `ci.yml` already has the commands to copy.
4. Log in to GHCR with `GITHUB_TOKEN` and push `ghcr.io/${owner,,}/${repo,,}:<tag>`. The name must be lowercase.
5. `gh release create <tag>` with `metrics.json` attached and the CHANGELOG section as the notes.

## Step 2: Ship v0.1 (owner: ___)
1. Merge the scaffold PR and the release workflow PR.
2. `git tag v0.1 <merge-commit> && git push origin v0.1`.
3. On GitHub, open Packages → the image → Package settings → **Change visibility → Public**. It is private by default, and if you skip this step every acceptance test fails.
4. Verify it the way the grader will:
   `docker logout ghcr.io && docker pull ghcr.io/<ns>/<repo>:v0.1 && docker run -p 80:80 ...` and run the three curls from the brief.

## Step 3: v0.2 experiment (`feat/v0.2-model`, owner: ___)
1. Compare candidates on **`cv_rmse_mean`**, the 5-fold cross-validated RMSE on the training split that `ml.train` already computes: Ridge (tune alpha with `RidgeCV`), Lasso, RandomForest, GradientBoosting. Log each to MLflow; hyperparameters are logged automatically from `get_params()`. Our MLflow use rests on this comparison, so screenshot the comparison view.
2. **Do not pick the winner by test `rmse`.** With n_test = 89, choosing on the test set leaks it into model selection and overstates the result. Pick on `cv_rmse_mean` (look at `cv_rmse_std` too: a gap smaller than the std is noise), then report the test `rmse` of the chosen model once. Expect only small gains. Linear models usually win on this dataset, and RandomForest is often *worse*. Pick the honest winner, even if the gain is small.
3. Optional but strong: add a "high-risk" flag (for example, predicted progression ≥ the 75th percentile of y_train) and report precision/recall. Keep the `/predict` response **exactly** `{"prediction": float}`. If you want the flag in the API, use a separate field only after checking it doesn't break the grader. The safer option is to report it only in CHANGELOG/metrics.json.
4. Change `build_pipeline()` in `ml/train.py`, set `MODEL_VERSION = "v0.2"`, and fill in the v0.2 section of CHANGELOG.md with side-by-side metrics and a rationale.

## Step 4: Ship v0.2 (owner: ___)
Merge, then `git tag v0.2 && git push origin v0.2`. After that, re-check that **both** `:v0.1` and `:v0.2` pull anonymously and report the right `model_version`.

## Step 5: Polish (anyone, small PRs)
- Add the real GHCR image names to the README
- Optional: measure p50 `/predict` latency and image size per version, and add them to CHANGELOG
- Hand-in: a `.txt` file containing the public repo URL
