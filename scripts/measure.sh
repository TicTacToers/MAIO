#!/usr/bin/env bash
# Measures the two numbers the CHANGELOG compares between versions:
#   1. image size (MB)
#   2. /predict response time (median and p95, in milliseconds)
#
# Usage:
#   scripts/measure.sh                              # build the image from the current code
#   scripts/measure.sh ghcr.io/tictactoers/maio:v0.1  # measure an already published image
#   N=500 scripts/measure.sh                        # change the number of timed requests
#
# Every run sends the same requests in the same order, so numbers from
# different versions can be compared. Run both versions on the same computer.
# Works with the bash that ships with macOS (3.2) and on Linux.
set -euo pipefail

IMAGE="${1:-}"
N="${N:-200}"           # timed requests
WARMUP="${WARMUP:-20}"  # untimed requests first, so start-up effects are not measured
PORT="${PORT:-8000}"    # host port; 8000 so it does not clash with a container on port 80
NAME="maio-measure"
URL="http://localhost:$PORT"

if [ -z "$IMAGE" ]; then
  IMAGE="maio:measure"
  echo "Building $IMAGE from the current code..."
  docker build -q -t "$IMAGE" . > /dev/null
elif ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "Pulling $IMAGE..."
  docker pull -q "$IMAGE" > /dev/null
fi

# 1. Image size
bytes=$(docker image inspect "$IMAGE" --format '{{.Size}}')
size_mb=$(awk -v b="$bytes" 'BEGIN { printf "%.1f", b / 1000000 }')

# Start the container and always remove it again, even if the script fails.
docker rm -f "$NAME" > /dev/null 2>&1 || true
trap 'docker rm -f "$NAME" > /dev/null 2>&1 || true' EXIT
docker run -d --name "$NAME" -p "$PORT:80" "$IMAGE" > /dev/null

for _ in $(seq 1 30); do
  curl -sf "$URL/health" > /dev/null && break
  sleep 1
done
version=$(curl -sf "$URL/health" | sed -n 's/.*"model_version": *"\([^"]*\)".*/\1/p') || true
[ -n "$version" ] || { echo "Service did not start"; docker logs "$NAME"; exit 1; }

# 2. Response time of /predict (same request every time)
payload='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
predict() {
  curl -s -o /dev/null -w '%{time_total}\n' -X POST "$URL/predict" \
    -H "Content-Type: application/json" -d "$payload"
}
for _ in $(seq 1 "$WARMUP"); do predict > /dev/null; done
times=$(for _ in $(seq 1 "$N"); do predict; done | sort -n)

# Median and 95th percentile, converted from seconds to milliseconds
median_ms=$(echo "$times" | awk '{ t[NR] = $1 } END {
  m = (NR % 2) ? t[(NR + 1) / 2] : (t[NR / 2] + t[NR / 2 + 1]) / 2
  printf "%.2f", m * 1000 }')
p95_ms=$(echo "$times" | awk '{ t[NR] = $1 } END {
  i = int(NR * 0.95 + 0.999999); printf "%.2f", t[i] * 1000 }')

echo ""
echo "Image:          $IMAGE"
echo "Model version:  $version"
echo "Image size:     $size_mb MB"
echo "/predict:       median $median_ms ms, p95 $p95_ms ms ($N requests after $WARMUP warm-up)"
echo ""
echo "For CHANGELOG.md:"
echo "| Image size           | $size_mb MB |"
echo "| p50 /predict latency | $median_ms ms |"
