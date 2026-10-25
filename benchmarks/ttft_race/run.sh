#!/usr/bin/env bash
# Sequential TTFT race on one GPU host. Each side owns the whole GPU while it is
# measured: Mem0, then LMCache CacheBlend, then InferScale. One question's three
# answers are then replayed side by side as ${BENCHMARK_RESULTS_ROOT}/ttft-race/ttft_race.gif.
set -euo pipefail
if (($#)); then
  echo "This script takes no arguments; run benchmarks.ttft_race.run for other settings." >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/environment.sh
source "${SCRIPT_DIR}/../../scripts/environment.sh"
cd "${PROJECT_ROOT}"
load_benchmark_environment "${PROJECT_ROOT}/configs/runtime.json" runtime
PYTHON="$(benchmark_python)"
export PATH="${VENV_DIR}/bin:${PATH}"
OUT="${BENCHMARK_RESULTS_ROOT}/ttft-race"

"${PYTHON}" -m benchmarks.ttft_race.run measure --mode mem0 --out "${OUT}/mem0.json"
"${PYTHON}" -m benchmarks.ttft_race.run measure --mode lmcache --out "${OUT}/lmcache.json"
"${PYTHON}" -m benchmarks.ttft_race.run measure --mode inferscale --out "${OUT}/inferscale.json"
"${PYTHON}" -m benchmarks.ttft_race.run gif "${OUT}/inferscale.json" "${OUT}/mem0.json" "${OUT}/lmcache.json" --out "${OUT}/ttft_race.gif"
