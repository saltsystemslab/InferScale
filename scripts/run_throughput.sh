#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

declare -a PASSTHROUGH_ARGS=("__throughput_sentinel__")
for argument in "$@"; do
  case "${argument}" in
    MODEL=*) export MODEL="${argument#MODEL=}" ;;
    CONDITIONS=*) export CONDITIONS="${argument#CONDITIONS=}" ;;
    RUN_ID=*) export RUN_ID="${argument#RUN_ID=}" ;;
    RESULTS_DIR=*) export RESULTS_DIR="${argument#RESULTS_DIR=}" ;;
    MATRIX=*) export MATRIX="${argument#MATRIX=}" ;;
    DRY_RUN=*) export DRY_RUN="${argument#DRY_RUN=}" ;;
    *) PASSTHROUGH_ARGS+=("${argument}") ;;
  esac
done

MODEL="${MODEL:-llama}"
CONDITIONS="${CONDITIONS:-no_memory prompt_injection kv_injection mem0}"
MATRIX="${MATRIX:-10:512,10:1024,10:2048,10:4096,25:512,25:1024,25:2048,25:4096,50:512,50:1024,50:2048,50:4096,100:512,100:1024,100:2048}"
RESULTS_DIR="${RESULTS_DIR:-${BENCHMARK_RESULTS_ROOT}}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_SLUG="$(printf '%s' "${MODEL}" | tr '/: ' '---' | tr '[:upper:]' '[:lower:]')"
RUN_ID="${RUN_ID:-${MODEL_SLUG}-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

if command -v locomo-throughput-bench >/dev/null 2>&1; then
  BENCH_COMMAND=(locomo-throughput-bench)
else
  export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  BENCH_COMMAND=(python -m locomo_jasper_bench.throughput.run)
fi

read -r -a CONDITION_ARGS <<<"${CONDITIONS}"
COMMAND=(
  "${BENCH_COMMAND[@]}"
  --model "${MODEL}"
  --conditions "${CONDITION_ARGS[@]}"
  --matrix "${MATRIX}"
  --results-dir "${RESULTS_DIR}"
  --run-id "${RUN_ID}"
  --requests-per-user "${REQUESTS_PER_USER:-2}"
  --max-output-tokens "${MAX_OUTPUT_TOKENS:-50}"
  --warmup-batches "${WARMUP_BATCHES:-2}"
  --top-k "${TOP_K:-10}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.52}"
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --kv-max-position "${KV_MAX_POSITION:-32768}"
  --kv-block-size "${KV_BLOCK_SIZE:-16}"
  --vector-backend "${VECTOR_BACKEND:-qdrant}"
  "${PASSTHROUGH_ARGS[@]:1}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  COMMAND+=(--dry-run)
  printf 'Running: '
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exec "${COMMAND[@]}"
fi

LOG_FILE="${LOG_FILE:-${RESULTS_DIR}/throughput/${RUN_ID}/run.log}"
mkdir -p "$(dirname -- "${LOG_FILE}")"
printf 'Running: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
