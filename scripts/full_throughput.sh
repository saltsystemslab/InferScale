#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

declare -a PASSTHROUGH_ARGS=("__throughput_sentinel__")
for argument in "$@"; do
  case "${argument}" in
    MODELS=*) export MODELS="${argument#MODELS=}" ;;
    CONDITIONS=*) export CONDITIONS="${argument#CONDITIONS=}" ;;
    RUN_ID=*) export RUN_ID="${argument#RUN_ID=}" ;;
    RESULTS_DIR=*) export RESULTS_DIR="${argument#RESULTS_DIR=}" ;;
    USER_COUNTS=*) export USER_COUNTS="${argument#USER_COUNTS=}" ;;
    DATASET=*) export DATASET="${argument#DATASET=}" ;;
    DRY_RUN=*) export DRY_RUN="${argument#DRY_RUN=}" ;;
    *) PASSTHROUGH_ARGS+=("${argument}") ;;
  esac
done

MODELS="${MODELS:-llama mistral qwen qwen3-14b}"
CONDITIONS="${CONDITIONS:-mem0_qdrant kv_injection}"
RESULTS_DIR="${RESULTS_DIR:-${BENCHMARK_RESULTS_ROOT}}"
DRY_RUN="${DRY_RUN:-0}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_PREFIX="${RUN_ID:-throughput-${RUN_STAMP}}"
LOG_DIR="${RESULTS_DIR}/throughput-logs-${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

read -r -a MODEL_ARGS <<<"${MODELS}"
declare -a FAILURES=()

for model in "${MODEL_ARGS[@]}"; do
  model_slug="$(printf '%s' "${model}" | tr '/: ' '---' | tr '[:upper:]' '[:lower:]')"
  model_run_id="${RUN_PREFIX}-${model_slug}"
  log_file="${LOG_DIR}/${model_slug}.log"
  printf '\n[%s] run_id=%s\n' "${model}" "${model_run_id}"
  MODEL="${model}" \
  CONDITIONS="${CONDITIONS}" \
  RUN_ID="${model_run_id}" \
  RESULTS_DIR="${RESULTS_DIR}" \
  DRY_RUN="${DRY_RUN}" \
  LOG_FILE="${log_file}" \
    "${SCRIPT_DIR}/run_throughput.sh" "${PASSTHROUGH_ARGS[@]:1}"
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    FAILURES+=("${model} (exit ${status})")
  fi
done

if (( ${#FAILURES[@]} )); then
  printf '\nFailed model runs:\n'
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '\nDry run complete.\n'
else
  printf '\nAll model runs completed. Logs are in %s\n' "${LOG_DIR}"
fi
