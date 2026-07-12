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
    USER_COUNTS=*) export USER_COUNTS="${argument#USER_COUNTS=}" ;;
    DATASET=*) export DATASET="${argument#DATASET=}" ;;
    CONTEXT_WINDOW=*) export CONTEXT_WINDOW="${argument#CONTEXT_WINDOW=}" ;;
    DRY_RUN=*) export DRY_RUN="${argument#DRY_RUN=}" ;;
    *) PASSTHROUGH_ARGS+=("${argument}") ;;
  esac
done

MODEL="${MODEL:-llama}"
CONDITIONS="${CONDITIONS:-no_memory mem0_qdrant mem0_jasper kv_injection}"
USER_COUNTS="${USER_COUNTS:-10,25,50,100}"
DATASET="${DATASET:-data/locomo10.json}"
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
  --user-counts "${USER_COUNTS}"
  --dataset "${DATASET}"
  --results-dir "${RESULTS_DIR}"
  --run-id "${RUN_ID}"
  --requests-per-user "${REQUESTS_PER_USER:-2}"
  --max-output-tokens "${MAX_OUTPUT_TOKENS:-50}"
  --warmup-batches "${WARMUP_BATCHES:-2}"
  --top-k "${TOP_K:-10}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.38}"
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --kv-max-position "${KV_MAX_POSITION:-32768}"
  --kv-block-size "${KV_BLOCK_SIZE:-16}"
  --context-window "${CONTEXT_WINDOW:-0}"
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
