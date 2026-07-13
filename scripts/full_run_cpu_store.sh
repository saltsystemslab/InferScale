#!/usr/bin/env bash
#
# Sweep the LoCoMo Jasper benchmark with the pinned-host (cpu-pinned) KV
# memory store. Same grid as scripts/full_run.sh but ONLY the vllm-kv+jasper
# runs: the vllm-prefix baselines never touch the KV store, so they come from
# the standard sweep.
#   models  = {llama, mistral, qwen, qwen3-14b}
#   top-k   = {5, 10, 20, 50, 100}
#   context window = {0, 5, 20, 50} turns preceding each retrieved fact
# => 4 models x 5 top-k x 4 windows = 80 runs.
#
# Usage:
#   BENCHMARK_RESULTS_ROOT=/path/to/results bash scripts/full_run_cpu_store.sh
#
#   # Preview every command without executing (recommended first):
#   DRY_RUN=1 BENCHMARK_RESULTS_ROOT=/path/to/results bash scripts/full_run_cpu_store.sh
#
#   # Override any grid, or the staging pool:
#   MODELS="llama" TOPKS="10 50" KV_WINDOWS="0 20" KV_STAGING_SLOTS=8 \
#     BENCHMARK_RESULTS_ROOT=/path bash scripts/full_run_cpu_store.sh
#
# Run IDs use the kvcpu marker so they never collide with gpu-store runs that
# share a stamp:
#   <model>-kvcpu-mem0-jasper10-k<topk>-s<W>-<stamp>

set -uo pipefail

# ----- Config (override via environment) -----------------------------------
MODELS="${MODELS:-llama mistral qwen qwen3-14b}"
TOPKS="${TOPKS:-5 10 20 50 100}"
KV_WINDOWS="${KV_WINDOWS:-${WINDOWS:-0 5 20 50}}"
KV_STAGING_SLOTS="${KV_STAGING_SLOTS:-4}"
DATASET="${DATASET:-data/locomo10.json}"
DRY_RUN="${DRY_RUN:-0}"

: "${BENCHMARK_RESULTS_ROOT:?Set BENCHMARK_RESULTS_ROOT to the results output directory}"

# ----- Preembed guard --------------------------------------------------------
# Every run consumes immutable Mem0 fact catalogs extracted with the answer
# model itself; fail once, up front, with the fix.
if [[ "${DRY_RUN}" != "1" ]]; then
  missing_models=()
  for MODEL in ${MODELS}; do
    if ! locomo-jasper-bench \
        --check-catalogs \
        --dataset "${DATASET}" \
        --answer-model "${MODEL}" \
        --max-samples 10 \
        --skip-judge; then
      missing_models+=("${MODEL}")
    fi
  done
  if (( ${#missing_models[@]} )); then
    {
      echo "ERROR: Mem0 fact catalogs are missing or incomplete for model(s): ${missing_models[*]}"
      echo "Extraction always uses the answer model; materialize the missing catalogs with:"
      echo "  EXTRACTION_MODELS=\"${missing_models[*]}\" bash scripts/extract_facts.sh"
    } >&2
    exit 1
  fi
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${BENCHMARK_RESULTS_ROOT}/sweep-cpu-logs-${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

# ----- Progress + failure tracking -----------------------------------------
n_models=$(wc -w <<<"${MODELS}")
n_topks=$(wc -w <<<"${TOPKS}")
n_windows=$(wc -w <<<"${KV_WINDOWS}")
TOTAL=$(( n_models * n_topks * n_windows ))
idx=0
declare -a FAILURES=()

# run_one <label> <command...>
run_one () {
  local label="$1"; shift
  idx=$(( idx + 1 ))
  printf '\n[%d/%d] %s\n      ' "${idx}" "${TOTAL}" "${label}"
  printf '%q ' "$@"; printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  local safe log rc
  safe=$(printf '%s' "${label}" | tr ' /=' '___')
  log="${LOG_DIR}/${safe}.log"

  "$@" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  if [[ "${rc}" -eq 0 ]]; then
    printf '      OK\n'
  else
    printf '      FAILED (exit %d) -- see %s\n' "${rc}" "${log}"
    FAILURES+=("${label} (exit ${rc})")
  fi
}

# ----- Sweep ----------------------------------------------------------------
for MODEL in ${MODELS}; do
  for TOP_K in ${TOPKS}; do
    for W in ${KV_WINDOWS}; do
      kv_id="${MODEL}-kvcpu-mem0-jasper10-k${TOP_K}-s${W}-${RUN_STAMP}"
      run_one "${MODEL} k=${TOP_K} s=${W} mem0-kvcpu jasper" \
          locomo-jasper-bench \
            --dataset "${DATASET}" \
            --results-dir "${BENCHMARK_RESULTS_ROOT}" \
            --answer-model "${MODEL}" \
            --answer-backend vllm-kv \
            --vector-backend jasper \
            --kv-store-backend cpu-pinned \
            --kv-staging-slots "${KV_STAGING_SLOTS}" \
            --top-k "${TOP_K}" \
            --context-window "${W}" \
            --kv-gpu-memory-utilization 0.30 \
            --max-samples 10 \
            --log-every 1 \
            --skip-judge \
            --run-id "${kv_id}"
    done
  done
done

# ----- Summary --------------------------------------------------------------
printf '\n===== Sweep complete: %d runs (stamp %s) =====\n' "${TOTAL}" "${RUN_STAMP}"
if (( ${#FAILURES[@]} )); then
  printf 'Failures (%d):\n' "${#FAILURES[@]}"
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
elif [[ "${DRY_RUN}" == "1" ]]; then
  printf '(dry run -- nothing executed)\n'
else
  printf 'All runs succeeded. Logs in %s\n' "${LOG_DIR}"
fi
