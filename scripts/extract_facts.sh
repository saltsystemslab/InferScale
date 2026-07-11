#!/usr/bin/env bash
#
# Materialize the immutable Mem0 fact catalogs for every answer model.
#
# For each model this script starts a local vLLM OpenAI-compatible server that
# serves THE ANSWER MODEL ITSELF, waits until it is healthy, runs the
# --preembed-only extraction phase against it, and shuts the server down before
# moving to the next model. Extraction therefore always uses the same model
# that later answers the benchmark questions.
#
# Usage:
#   bash scripts/extract_facts.sh
#   EXTRACTION_MODELS="llama" bash scripts/extract_facts.sh
#
# Env knobs:
#   EXTRACTION_MODELS                  answer models to extract for (default: llama mistral qwen qwen3-14b)
#   MEM0_LLM_PORT                      port for the extraction server (default: 8000)
#   EXTRACTION_GPU_MEMORY_UTILIZATION  vllm serve GPU fraction (default: 0.85)
#   EXTRACTION_MAX_MODEL_LEN           vllm serve context length (default: 32768)
#   EXTRACTION_HEALTH_TIMEOUT          seconds to wait for server health (default: 900)
#   EXTRACTION_EXTRA_VLLM_ARGS         extra args appended to vllm serve
#   DATASET                            LoCoMo dataset path (default: data/locomo10.json)
#   MAX_SAMPLES                        samples to extract (default: 10)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/load_env.sh
source "${SCRIPT_DIR}/load_env.sh"
# shellcheck source=scripts/vllm_env.sh
source "${SCRIPT_DIR}/vllm_env.sh"

if [[ -z "${VIRTUAL_ENV:-}" && -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

EXTRACTION_MODELS="${EXTRACTION_MODELS:-llama mistral qwen qwen3-14b}"
MEM0_LLM_PORT="${MEM0_LLM_PORT:-8000}"
EXTRACTION_GPU_MEMORY_UTILIZATION="${EXTRACTION_GPU_MEMORY_UTILIZATION:-0.85}"
EXTRACTION_MAX_MODEL_LEN="${EXTRACTION_MAX_MODEL_LEN:-32768}"
EXTRACTION_HEALTH_TIMEOUT="${EXTRACTION_HEALTH_TIMEOUT:-900}"
EXTRACTION_EXTRA_VLLM_ARGS="${EXTRACTION_EXTRA_VLLM_ARGS:-}"
DATASET="${DATASET:-data/locomo10.json}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"

: "${BENCHMARK_RESULTS_ROOT:?Set BENCHMARK_RESULTS_ROOT (normally exported by scripts/load_env.sh)}"

EXTRACTION_BASE_URL="http://localhost:${MEM0_LLM_PORT}/v1"
LOG_DIR="${TMPDIR:-/tmp}/extract-facts-logs"
mkdir -p "${LOG_DIR}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

SERVER_PID=""

stop_server() {
  local pid="${SERVER_PID}"
  SERVER_PID=""
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  # The server is started with setsid when available, so its PID is also its
  # process-group id; signal the group to take the vLLM workers down with it.
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 1
  done
  echo "warning: extraction vLLM server (pid ${pid}) ignored SIGTERM; sending SIGKILL." >&2
  kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}
trap stop_server EXIT

resolve_model() {
  python - "$1" <<'PY'
import sys

from locomo_jasper_bench.config import resolve_answer_model

print(resolve_answer_model(sys.argv[1]))
PY
}

resolve_reasoning_parser() {
  python - "$1" <<'PY'
import sys

from locomo_jasper_bench.config import resolve_reasoning_parser

print(resolve_reasoning_parser(sys.argv[1]) or "")
PY
}

wait_for_server() {
  local log_file="$1"
  local waited=0
  while (( waited < EXTRACTION_HEALTH_TIMEOUT )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: extraction vLLM server exited before becoming healthy. Log tail:" >&2
      tail -n 40 "${log_file}" >&2
      return 1
    fi
    if curl -sf "${EXTRACTION_BASE_URL}/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
  echo "ERROR: extraction vLLM server was not healthy after ${EXTRACTION_HEALTH_TIMEOUT}s. Log tail:" >&2
  tail -n 40 "${log_file}" >&2
  return 1
}

if curl -sf "${EXTRACTION_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: something is already serving on ${EXTRACTION_BASE_URL} (a judge server?)." >&2
  echo "Stop it or set MEM0_LLM_PORT to a free port before extracting facts." >&2
  exit 1
fi

for MODEL_ALIAS in ${EXTRACTION_MODELS}; do
  RESOLVED_MODEL="$(resolve_model "${MODEL_ALIAS}")"
  MODEL_SLUG="$(printf '%s' "${MODEL_ALIAS}" | tr ' /' '__')"
  LOG_FILE="${LOG_DIR}/vllm-${MODEL_SLUG}-${RUN_STAMP}.log"

  SERVE_ARGS=(
    --port "${MEM0_LLM_PORT}"
    --trust-remote-code
    --dtype auto
    --max-model-len "${EXTRACTION_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${EXTRACTION_GPU_MEMORY_UTILIZATION}"
  )
  # Reasoning models emit thinking tokens; the parser keeps them out of
  # message content so mem0's JSON extraction parsing sees only the answer.
  # Resolved per alias in config.py so overridden checkpoints keep the parser.
  REASONING_PARSER="$(resolve_reasoning_parser "${MODEL_ALIAS}")"
  if [[ -n "${REASONING_PARSER}" ]]; then
    SERVE_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
  fi
  if [[ -n "${EXTRACTION_EXTRA_VLLM_ARGS}" ]]; then
    # shellcheck disable=SC2206
    SERVE_ARGS+=(${EXTRACTION_EXTRA_VLLM_ARGS})
  fi

  echo "=== Extracting facts with ${MODEL_ALIAS} (${RESOLVED_MODEL}) ==="
  echo "Starting extraction vLLM server on port ${MEM0_LLM_PORT}; log: ${LOG_FILE}"
  if command -v setsid >/dev/null 2>&1; then
    setsid vllm serve "${RESOLVED_MODEL}" "${SERVE_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
  else
    vllm serve "${RESOLVED_MODEL}" "${SERVE_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
  fi
  SERVER_PID=$!

  wait_for_server "${LOG_FILE}"
  echo "Extraction server is healthy at ${EXTRACTION_BASE_URL}"

  MEM0_LLM_BASE_URL="${EXTRACTION_BASE_URL}" locomo-jasper-bench \
    --dataset "${DATASET}" \
    --results-dir "${BENCHMARK_RESULTS_ROOT}" \
    --answer-model "${MODEL_ALIAS}" \
    --vector-backend qdrant \
    --max-samples "${MAX_SAMPLES}" \
    --preembed-only \
    --run-id "setup-preembed-${MODEL_SLUG}-${RUN_STAMP}"

  stop_server
  # Make sure the port is actually free before the next model claims it.
  for _ in $(seq 1 30); do
    curl -sf "${EXTRACTION_BASE_URL}/models" >/dev/null 2>&1 || break
    sleep 1
  done
  if curl -sf "${EXTRACTION_BASE_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: port ${MEM0_LLM_PORT} is still serving after shutdown." >&2
    exit 1
  fi
done

echo "=== Fact extraction complete for: ${EXTRACTION_MODELS} ==="
