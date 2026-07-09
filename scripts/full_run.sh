#!/usr/bin/env bash
#
# Sweep the LoCoMo Jasper benchmark across:
#   models  = {llama, mistral, qwen, qwen3-14b}
#   top-k   = {5, 10, 20, 50, 100}
#   window  = {0, 1, 3, 5}   (applied to the kv-jasper run ONLY)
#
# Per (model, top-k) it runs:
#   - vllm-kv     + jasper, once per window W  (--context-window W)   -> 4 runs
#   - vllm-prefix + jasper, once (no window flag, as in your template) -> 1 run
#   - vllm-prefix + qdrant, once (no window flag, as in your template) -> 1 run
# => 4 models x 5 top-k x (4 + 1 + 1) = 120 runs.
#
# Usage:
#   BENCHMARK_RESULTS_ROOT=/path/to/results ./run_locomo_sweep.sh
#
#   # Preview every command without executing (recommended first):
#   DRY_RUN=1 BENCHMARK_RESULTS_ROOT=/path/to/results ./run_locomo_sweep.sh
#
#   # Override any grid:
#   MODELS="llama qwen3-14b" TOPKS="10 50" WINDOWS="0 3" BENCHMARK_RESULTS_ROOT=/path ./run_locomo_sweep.sh
#
# Run IDs encode the swept axes so results never collide:
#   kv     -> <model>-kv-gpu-jasper10-k<topk>-w<W>-<stamp>
#   prefix -> <model>-prefix-gpu-jasper10-k<topk>-<stamp>
#   qdrant -> <model>-prefix-qdrant10-k<topk>-<stamp>

set -uo pipefail

# ----- Config (override via environment) -----------------------------------
MODELS="${MODELS:-llama mistral qwen qwen3-14b}"
TOPKS="${TOPKS:-5 10 20 50 100}"
WINDOWS="${WINDOWS:-0 1 3 5}"
DATASET="${DATASET:-data/locomo10.json}"
DRY_RUN="${DRY_RUN:-0}"

: "${BENCHMARK_RESULTS_ROOT:?Set BENCHMARK_RESULTS_ROOT to the results output directory}"

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${BENCHMARK_RESULTS_ROOT}/sweep-logs-${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

# ----- Progress + failure tracking -----------------------------------------
n_models=$(wc -w <<<"${MODELS}")
n_topks=$(wc -w <<<"${TOPKS}")
n_windows=$(wc -w <<<"${WINDOWS}")
# kv runs once per window; the two prefix backends run once each per (model,top-k).
TOTAL=$(( n_models * n_topks * (n_windows + 2) ))
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

    # 1) vLLM KV cache + jasper -- swept over context-window W
    for W in ${WINDOWS}; do
      kv_id="${MODEL}-kv-gpu-jasper10-k${TOP_K}-w${W}-${RUN_STAMP}"
      run_one "${MODEL} k=${TOP_K} w=${W} kv-jasper" \
        locomo-jasper-bench \
          --dataset "${DATASET}" \
          --results-dir "${BENCHMARK_RESULTS_ROOT}" \
          --answer-model "${MODEL}" \
          --answer-backend vllm-kv \
          --vector-backend jasper \
          --top-k "${TOP_K}" \
          --context-window "${W}" \
          --kv-gpu-memory-utilization 0.52 \
          --max-samples 10 \
          --log-every 1 \
          --skip-judge \
          --run-id "${kv_id}"
    done

    # 2) vLLM prefix + jasper -- once, no window flag (as in original)
    prefix_id="${MODEL}-prefix-gpu-jasper10-k${TOP_K}-${RUN_STAMP}"
    run_one "${MODEL} k=${TOP_K} prefix-jasper" \
      locomo-jasper-bench \
        --dataset "${DATASET}" \
        --results-dir "${BENCHMARK_RESULTS_ROOT}" \
        --answer-model "${MODEL}" \
        --answer-backend vllm-prefix \
        --vector-backend jasper \
        --top-k "${TOP_K}" \
        --kv-gpu-memory-utilization 0.52 \
        --max-samples 10 \
        --log-every 1 \
        --skip-judge \
        --run-id "${prefix_id}"

    # 3) vLLM prefix + qdrant -- once, no window flag (as in original)
    qdrant_id="${MODEL}-prefix-qdrant10-k${TOP_K}-${RUN_STAMP}"
    run_one "${MODEL} k=${TOP_K} prefix-qdrant" \
      locomo-jasper-bench \
        --dataset "${DATASET}" \
        --results-dir "${BENCHMARK_RESULTS_ROOT}" \
        --answer-model "${MODEL}" \
        --answer-backend vllm-prefix \
        --vector-backend qdrant \
        --top-k "${TOP_K}" \
        --kv-gpu-memory-utilization 0.52 \
        --max-samples 10 \
        --log-every 1 \
        --skip-judge \
        --run-id "${qdrant_id}"
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
