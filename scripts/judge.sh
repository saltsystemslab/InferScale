#!/usr/bin/env bash
#
# Judge all LoCoMo Jasper results for a given run stamp.
#
# The sweep ran with --skip-judge, so this runs the judge pass (--judge-only)
# over every run-id produced under a stamp (default: 20260706T073907Z).
#
# Enumerating run-ids (set RUNIDS_FROM):
#   discover (default) : scan BENCHMARK_RESULTS_ROOT for run-ids whose name contains STAMP.
#                        Judges exactly what exists -- skips sweep runs that produced nothing.
#   grid               : regenerate them from the current Mem0 fact sweep grid.
#                        Use this if discovery finds nothing due to the results layout.
#
# Required env: BENCHMARK_RESULTS_ROOT  JUDGE_BASE_URL  JUDGE_API_KEY  JUDGE_MODEL
# Optional:     STAMP  RUNIDS_FROM  DRY_RUN  MODELS/TOPKS/KV_WINDOWS
#
# Usage:
#   BENCHMARK_RESULTS_ROOT=/r JUDGE_BASE_URL=... JUDGE_API_KEY=... JUDGE_MODEL=... bash scripts/judge.sh
#   DRY_RUN=1 ...            bash scripts/judge.sh   # preview run-id list + command shape
#   RUNIDS_FROM=grid ...     bash scripts/judge.sh   # rebuild ids from the grid instead

set -uo pipefail

STAMP="${STAMP:-<RUN_STAMP>}"  # e.g. 20260706T073907Z
RUNIDS_FROM="${RUNIDS_FROM:-discover}"
DRY_RUN="${DRY_RUN:-0}"

: "${BENCHMARK_RESULTS_ROOT:?Set BENCHMARK_RESULTS_ROOT to the results directory}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL}"
: "${JUDGE_API_KEY:?Set JUDGE_API_KEY}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL}"

# Grid -- only used when RUNIDS_FROM=grid. Must match the sweep that produced results.
MODELS="${MODELS:-llama mistral qwen qwen3-14b}"
TOPKS="${TOPKS:-5 10 20 50 100}"
KV_WINDOWS="${KV_WINDOWS:-${WINDOWS:-0 5 20 50}}"

# ----- Build list of run-ids -----------------------------------------------
declare -a RUN_IDS=()

if [[ "${RUNIDS_FROM}" == "grid" ]]; then
  for MODEL in ${MODELS}; do
    for TOP_K in ${TOPKS}; do
      for W in ${KV_WINDOWS}; do
        RUN_IDS+=("${MODEL}-kv-mem0-jasper10-k${TOP_K}-s${W}-${STAMP}")
      done
      RUN_IDS+=("${MODEL}-prefix-mem0-qdrant10-k${TOP_K}-s0-${STAMP}")
      RUN_IDS+=("${MODEL}-prefix-mem0-jasper10-k${TOP_K}-s0-${STAMP}")
    done
  done
elif [[ "${RUNIDS_FROM}" == "discover" ]]; then
  # Any path under results-dir mentioning the stamp -> pull out the run-id token.
  # Filter to tokens containing the tool's backend markers so result files that merely
  # embed the stamp (e.g. metrics-<stamp>.json) don't get mistaken for run-ids.
  while IFS= read -r RUN_ID; do
    RUN_IDS+=("${RUN_ID}")
  done < <(
    find "${BENCHMARK_RESULTS_ROOT}" -name "*${STAMP}*" 2>/dev/null \
      | grep -oE "[A-Za-z0-9]+(-[A-Za-z0-9]+)*-${STAMP}" \
      | grep -E -- "-(kv-mem0-(jasper|qdrant)10|kvcpu-mem0-jasper10|prefix-mem0-(jasper|qdrant)10|kv-gpu-jasper10|prefix-qdrant10)-" \
      | sort -u
  )
else
  echo "RUNIDS_FROM must be 'discover' or 'grid' (got: ${RUNIDS_FROM})" >&2
  exit 2
fi

if (( ${#RUN_IDS[@]} == 0 )); then
  echo "No run-ids found for stamp ${STAMP}." >&2
  if [[ "${RUNIDS_FROM}" == "discover" ]]; then
    echo "Nothing under ${BENCHMARK_RESULTS_ROOT} matched. If results exist but are named" >&2
    echo "differently, retry with: RUNIDS_FROM=grid $0" >&2
  fi
  exit 1
fi

TOTAL=${#RUN_IDS[@]}

# ----- Dry run: show what would happen, then stop --------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Would judge ${TOTAL} run(s) for stamp ${STAMP} (source: ${RUNIDS_FROM})."
  echo
  echo "Command per run-id (API key redacted):"
  echo "  locomo-jasper-bench --results-dir ${BENCHMARK_RESULTS_ROOT} \\"
  echo "    --run-id <RUN_ID> --judge-only --judge vllm \\"
  echo "    --judge-base-url ${JUDGE_BASE_URL} --judge-api-key **** --judge-model ${JUDGE_MODEL}"
  echo
  echo "Run-ids:"
  printf '  %s\n' "${RUN_IDS[@]}"
  echo
  echo "(dry run -- nothing executed)"
  exit 0
fi

# ----- Judge each -----------------------------------------------------------
echo "Judging ${TOTAL} run(s) for stamp ${STAMP} (source: ${RUNIDS_FROM})."
LOG_DIR="${BENCHMARK_RESULTS_ROOT}/judge-logs-${STAMP}"
mkdir -p "${LOG_DIR}"
idx=0
declare -a FAILURES=()

for RUN_ID in "${RUN_IDS[@]}"; do
  idx=$(( idx + 1 ))
  printf '\n[%d/%d] judging %s\n' "${idx}" "${TOTAL}" "${RUN_ID}"
  log="${LOG_DIR}/${RUN_ID}.judge.log"
  locomo-jasper-bench \
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

# ----- Summary --------------------------------------------------------------
printf '\n===== Judged %d run(s) for stamp %s =====\n' "${TOTAL}" "${STAMP}"
if (( ${#FAILURES[@]} )); then
  printf 'Failures (%d):\n' "${#FAILURES[@]}"
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
fi
printf 'All judged successfully. Logs in %s\n' "${LOG_DIR}"
