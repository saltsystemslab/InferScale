#!/usr/bin/env bash
#
# Sweep the RAG benchmark across MODELS x TOPKS x modes {vllm-kv, vllm-prefix}
# with --skip-judge (judge afterwards with scripts/rag/judge.sh).
#
# Usage:
#   BENCHMARK_RESULTS_ROOT=/path bash scripts/rag/full_run.sh
#   DRY_RUN=1 BENCHMARK_RESULTS_ROOT=/path bash scripts/rag/full_run.sh
#   MODELS="llama" TOPKS="3 5 10 15 20" BENCHMARK_RESULTS_ROOT=/path bash scripts/rag/full_run.sh
#
# Run IDs encode the swept axes so results never collide:
#   <model>-kv-<dataset>-c<chunk>-w<window>-k<topk>-<stamp>
#   <model>-prefix-<dataset>-c<chunk>-w<window>-k<topk>-<stamp>

set -uo pipefail

MODELS="${MODELS:-llama}"
TOPKS="${TOPKS:-15}"
RAG_WINDOW="${RAG_WINDOW:-5}"
CHUNK_SIZE="${RAG_CHUNK_SIZE:-1024}"
DATASET_NAME="${RAG_DATASET:-multihoprag}"
DRY_RUN="${DRY_RUN:-0}"

: "${BENCHMARK_RESULTS_ROOT:?Set BENCHMARK_RESULTS_ROOT to the results output directory}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${BENCHMARK_RESULTS_ROOT}/rag-sweep-logs-${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

n_models=$(wc -w <<<"${MODELS}")
n_topks=$(wc -w <<<"${TOPKS}")
TOTAL=$(( n_models * n_topks * 2 ))
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

for MODEL in ${MODELS}; do
  for TOP_K in ${TOPKS}; do
    kv_id="${MODEL}-kv-${DATASET_NAME}-c${CHUNK_SIZE}-w${RAG_WINDOW}-k${TOP_K}-${RUN_STAMP}"
    run_one "${MODEL} k=${TOP_K} rag-kv" \
      rag-jasper-bench \
        --dataset-name "${DATASET_NAME}" \
        --results-dir "${BENCHMARK_RESULTS_ROOT}" \
        --answer-model "${MODEL}" \
        --answer-backend vllm-kv \
        --chunk-size "${CHUNK_SIZE}" \
        --context-window "${RAG_WINDOW}" \
        --top-k "${TOP_K}" \
        --skip-judge \
        --run-id "${kv_id}"

    prefix_id="${MODEL}-prefix-${DATASET_NAME}-c${CHUNK_SIZE}-w${RAG_WINDOW}-k${TOP_K}-${RUN_STAMP}"
    run_one "${MODEL} k=${TOP_K} rag-prefix" \
      rag-jasper-bench \
        --dataset-name "${DATASET_NAME}" \
        --results-dir "${BENCHMARK_RESULTS_ROOT}" \
        --answer-model "${MODEL}" \
        --answer-backend vllm-prefix \
        --chunk-size "${CHUNK_SIZE}" \
        --context-window "${RAG_WINDOW}" \
        --top-k "${TOP_K}" \
        --skip-judge \
        --run-id "${prefix_id}"
  done
done

printf '\n===== RAG sweep complete: %d runs (stamp %s) =====\n' "${TOTAL}" "${RUN_STAMP}"
if (( ${#FAILURES[@]} )); then
  printf 'Failures (%d):\n' "${#FAILURES[@]}"
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
elif [[ "${DRY_RUN}" == "1" ]]; then
  printf '(dry run -- nothing executed)\n'
else
  printf 'All runs succeeded. Logs in %s\n' "${LOG_DIR}"
fi
