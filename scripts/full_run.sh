#!/usr/bin/env bash
#
# Sweep the LoCoMo Jasper benchmark across:
#   models  = {llama, mistral, qwen, qwen3-14b}
#   top-k   = {5, 10, 20, 50}
#   context window = {0, 5, 20, 50} turns preceding each retrieved fact
#
# Per (model, top-k) it runs:
#   - vllm-kv     + jasper, once per window W (--context-window W) -> 4 runs
#   - vllm-prefix + qdrant, once as a separate baseline           -> 1 run
#   - vllm-prefix + jasper, once as a separate baseline           -> 1 run
# => 4 models x 4 top-k x (4 + 2) = 96 runs.
#
# Usage:
#   BENCHMARK_RESULTS_ROOT=/path/to/results bash scripts/full_run.sh
#
#   # Preview every command without executing (recommended first):
#   DRY_RUN=1 BENCHMARK_RESULTS_ROOT=/path/to/results bash scripts/full_run.sh
#
#   # Override any grid:
#   MODELS="llama qwen3-14b" TOPKS="10 50" KV_WINDOWS="0 20" \
#     BENCHMARK_RESULTS_ROOT=/path bash scripts/full_run.sh
#
# Run IDs encode the swept axes so results never collide:
#   KV     -> <model>-kv-mem0-<vector>10-k<topk>-s<W>-<stamp>
#   prefix -> <model>-prefix-mem0-<vector>10-k<topk>-s<W>-<stamp>

set -uo pipefail

# ----- Config (override via environment) -----------------------------------
MODELS="${MODELS:-llama mistral qwen qwen3-14b}"
TOPKS="${TOPKS:-5 10 20 50}"
KV_WINDOWS="${KV_WINDOWS:-${WINDOWS:-0 5 20 50}}"
DATASET="${DATASET:-data/locomo10.json}"
DRY_RUN="${DRY_RUN:-0}"

: "${BENCHMARK_RESULTS_ROOT:?Set BENCHMARK_RESULTS_ROOT to the results output directory}"

# ----- Preembed guard --------------------------------------------------------
# Every run consumes immutable Mem0 fact catalogs extracted with the answer
# model itself, so each swept model needs its own catalogs. Without them all of
# that model's runs fail identically at their first sample. Fail once, up
# front, with the fix.
# The check uses the exact catalog identity the runs will use (model,
# endpoint, embedding, mem0 version, sample fingerprints).
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
LOG_DIR="${BENCHMARK_RESULTS_ROOT}/sweep-logs-${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

# ----- Progress + failure tracking -----------------------------------------
n_models=$(wc -w <<<"${MODELS}")
n_topks=$(wc -w <<<"${TOPKS}")
n_windows=$(wc -w <<<"${KV_WINDOWS}")
# Per (model, topk): one KV run per window plus the single prefix-qdrant
# prompt-injection baseline.
TOTAL=$(( n_models * n_topks * (n_windows + 1) ))
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
      kv_id="${MODEL}-kv-mem0-jasper10-k${TOP_K}-s${W}-${RUN_STAMP}"
      run_one "${MODEL} k=${TOP_K} s=${W} mem0-kv jasper" \
          locomo-jasper-bench \
            --dataset "${DATASET}" \
            --results-dir "${BENCHMARK_RESULTS_ROOT}" \
            --answer-model "${MODEL}" \
            --answer-backend vllm-kv \
            --vector-backend jasper \
            --top-k "${TOP_K}" \
            --context-window "${W}" \
            --kv-gpu-memory-utilization 0.30 \
            --max-samples 10 \
            --log-every 1 \
            --skip-judge \
            --run-id "${kv_id}"

    done

    prefix_id="${MODEL}-prefix-mem0-qdrant10-k${TOP_K}-s0-${RUN_STAMP}"
    run_one "${MODEL} k=${TOP_K} mem0-prefix qdrant" \
      locomo-jasper-bench \
        --dataset "${DATASET}" \
        --results-dir "${BENCHMARK_RESULTS_ROOT}" \
        --answer-model "${MODEL}" \
        --answer-backend vllm-prefix \
        --vector-backend qdrant \
        --top-k "${TOP_K}" \
        --context-window 0 \
        --kv-gpu-memory-utilization 0.30 \
        --max-samples 10 \
        --log-every 1 \
        --skip-judge \
        --run-id "${prefix_id}"
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
