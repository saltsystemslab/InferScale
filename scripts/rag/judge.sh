#!/usr/bin/env bash
#
# Judge all RAG benchmark results for a given run stamp (deferred judging).
#
# The sweep ran with --skip-judge; this runs rag-jasper-bench --judge-only over
# every RAG run-id produced under STAMP. The judge server is the unchanged
# scripts/serve_vllm.sh (Gemma). This script is separate from scripts/judge.sh
# on purpose: the LoCoMo discovery regex must not learn RAG run-id shapes.
#
# Enumerating run-ids (set RUNIDS_FROM):
#   discover (default) : scan BENCHMARK_RESULTS_ROOT for RAG run-ids containing STAMP.
#   grid               : regenerate them from the current sweep grid.
#
# Required env: STAMP  BENCHMARK_RESULTS_ROOT  JUDGE_BASE_URL  JUDGE_API_KEY  JUDGE_MODEL
# Optional:     RUNIDS_FROM  DRY_RUN  MODELS  TOPKS  RAG_WINDOW  RAG_CHUNK_SIZE  RAG_DATASET

set -uo pipefail

: "${STAMP:?Set STAMP to the sweep run stamp to judge, e.g. 20260730T120000Z}"
RUNIDS_FROM="${RUNIDS_FROM:-discover}"
DRY_RUN="${DRY_RUN:-0}"

: "${BENCHMARK_RESULTS_ROOT:?Set BENCHMARK_RESULTS_ROOT to the results directory}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL}"
: "${JUDGE_API_KEY:?Set JUDGE_API_KEY}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL}"

MODELS="${MODELS:-llama}"
TOPKS="${TOPKS:-15}"
RAG_WINDOW="${RAG_WINDOW:-5}"
CHUNK_SIZE="${RAG_CHUNK_SIZE:-1024}"
DATASET_NAME="${RAG_DATASET:-multihoprag}"

declare -a RUN_IDS=()

if [[ "${RUNIDS_FROM}" == "grid" ]]; then
  for MODEL in ${MODELS}; do
    for TOP_K in ${TOPKS}; do
      RUN_IDS+=("${MODEL}-kv-${DATASET_NAME}-c${CHUNK_SIZE}-w${RAG_WINDOW}-k${TOP_K}-${STAMP}")
      RUN_IDS+=("${MODEL}-prefix-${DATASET_NAME}-c${CHUNK_SIZE}-w${RAG_WINDOW}-k${TOP_K}-${STAMP}")
    done
  done
elif [[ "${RUNIDS_FROM}" == "discover" ]]; then
  while IFS= read -r RUN_ID; do
    RUN_IDS+=("${RUN_ID}")
  done < <(
    find "${BENCHMARK_RESULTS_ROOT}" -name "*${STAMP}*" 2>/dev/null \
      | grep -oE "[A-Za-z0-9]+(-[A-Za-z0-9]+)*-${STAMP}" \
      | grep -E -- "-(kv|prefix)-${DATASET_NAME}-c[0-9]+-w[0-9]+-k[0-9]+-${STAMP}$" \
      | sort -u
  )
else
  echo "RUNIDS_FROM must be 'discover' or 'grid' (got: ${RUNIDS_FROM})" >&2
  exit 2
fi

if (( ${#RUN_IDS[@]} == 0 )); then
  echo "No RAG run-ids found for stamp ${STAMP}." >&2
  if [[ "${RUNIDS_FROM}" == "discover" ]]; then
    echo "Nothing under ${BENCHMARK_RESULTS_ROOT} matched. If results exist but are named" >&2
    echo "differently, retry with: RUNIDS_FROM=grid $0" >&2
  fi
  exit 1
fi

TOTAL=${#RUN_IDS[@]}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Would judge ${TOTAL} RAG run(s) for stamp ${STAMP} (source: ${RUNIDS_FROM})."
  echo
  echo "Command per run-id (API key redacted):"
  echo "  rag-jasper-bench --results-dir ${BENCHMARK_RESULTS_ROOT} \\"
  echo "    --run-id <RUN_ID> --judge-only --judge vllm \\"
  echo "    --judge-base-url ${JUDGE_BASE_URL} --judge-api-key **** --judge-model ${JUDGE_MODEL}"
  echo
  echo "Run-ids:"
  printf '  %s\n' "${RUN_IDS[@]}"
  echo
  echo "(dry run -- nothing executed)"
  exit 0
fi

echo "Judging ${TOTAL} RAG run(s) for stamp ${STAMP} (source: ${RUNIDS_FROM})."
LOG_DIR="${BENCHMARK_RESULTS_ROOT}/rag-judge-logs-${STAMP}"
mkdir -p "${LOG_DIR}"
idx=0
declare -a FAILURES=()

for RUN_ID in "${RUN_IDS[@]}"; do
  idx=$(( idx + 1 ))
  printf '\n[%d/%d] judging %s\n' "${idx}" "${TOTAL}" "${RUN_ID}"
  log="${LOG_DIR}/${RUN_ID}.judge.log"
  rag-jasper-bench \
    --results-dir "${BENCHMARK_RESULTS_ROOT}" \
    --run-id "${RUN_ID}" \
    --judge-only \
    --judge vllm \
    --judge-base-url "${JUDGE_BASE_URL}" \
    --judge-api-key "${JUDGE_API_KEY}" \
    --judge-model "${JUDGE_MODEL}" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  if [[ "${rc}" -eq 0 ]]; then
    printf '      OK\n'
  else
    printf '      FAILED (exit %d) -- see %s\n' "${rc}" "${log}"
    FAILURES+=("${RUN_ID} (exit ${rc})")
  fi
done

printf '\n===== Judged %d RAG run(s) for stamp %s =====\n' "${TOTAL}" "${STAMP}"
if (( ${#FAILURES[@]} )); then
  printf 'Failures (%d):\n' "${#FAILURES[@]}"
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
fi
printf 'All judged successfully. Logs in %s\n' "${LOG_DIR}"
