#!/usr/bin/env bash
# Container smoke test: runs the same checks the graders run against a started container.
# Used by both ci.yml (every PR/push) and release.yml (every tag), so there is one copy.
#
# Usage: scripts/smoke.sh <expected_model_version> [base_url]
#   e.g. scripts/smoke.sh v0.1 http://localhost:80
set -euo pipefail

EXPECTED_VERSION="${1:?usage: smoke.sh <expected_model_version> [base_url]}"
BASE_URL="${2:-http://localhost:80}"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# 1. Wait up to 30 s for the service to start.
for _ in $(seq 1 30); do
  curl -sf "$BASE_URL/health" > /dev/null && break
  sleep 1
done

# 2. GET /health -> {"status":"ok","model_version":"<expected>"}
health=$(curl -sf "$BASE_URL/health") || fail "/health did not answer"
echo "/health -> $health"
[ "$(jq -r .status <<< "$health")" = "ok" ] || fail "status is not ok"
[ "$(jq -r .model_version <<< "$health")" = "$EXPECTED_VERSION" ] \
  || fail "model_version is not $EXPECTED_VERSION"

# 3. POST /predict with all ten features -> {"prediction": <number>}
valid='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
pred=$(curl -sf -X POST "$BASE_URL/predict" -H "Content-Type: application/json" -d "$valid") \
  || fail "/predict rejected a valid request"
echo "/predict -> $pred"
jq -e '.prediction | type == "number"' <<< "$pred" > /dev/null || fail "prediction is not a number"

# 4. The graders' invalid case: every feature except s6 -> HTTP 422 + JSON naming s6
missing_s6='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}'
code=$(curl -s -o /tmp/smoke_422.json -w '%{http_code}' -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" -d "$missing_s6")
echo "/predict without s6 -> HTTP $code $(cat /tmp/smoke_422.json)"
[ "$code" = "422" ] || fail "expected HTTP 422, got $code"
jq -e '.details[] | select(.field == "s6")' /tmp/smoke_422.json > /dev/null \
  || fail "422 body does not name the missing field s6"

echo "SMOKE OK: all checks passed for $EXPECTED_VERSION"
