#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${VIRTUAL_ENV:-}" && -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Model aliases resolve through config.py so precedence and alias tables have a
# single source of truth. Raw HF ids and local paths pass through unchanged.
resolve_model() {
  python - "$1" <<'PY'
import sys

from locomo_jasper_bench.config import resolve_answer_model

print(resolve_answer_model(sys.argv[1]))
PY
}

MODEL="$(resolve_model "${JUDGE_MODEL:-${LOCOMO_VLLM_MODEL:-Gemma-2-9B-Instruct}}")"
API_KEY="${JUDGE_API_KEY:-${LOCOMO_VLLM_API_KEY:-token-abc123}}"
TP="${LOCOMO_VLLM_TP:-1}"
GPU_MEMORY_UTILIZATION="${LOCOMO_VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
DTYPE="${LOCOMO_VLLM_DTYPE:-auto}"
QUANTIZATION="${LOCOMO_VLLM_QUANTIZATION:-}"

# shellcheck source=scripts/vllm_env.sh
source "${SCRIPT_DIR}/vllm_env.sh"

QUANTIZATION_ARGS=()
if [[ -n "${QUANTIZATION}" ]]; then
  QUANTIZATION_ARGS=(--quantization "${QUANTIZATION}")
fi

# Bound whitespace in schema-guided outputs so extraction models cannot loop
# on newlines until max_tokens truncates the JSON mid-structure. Override with
# LOCOMO_VLLM_STRUCTURED_OUTPUTS_CONFIG (set it empty to omit the flag; on
# older vLLM builds without --structured-outputs-config, disable it here and
# use --guided-decoding-backend xgrammar:disable-any-whitespace instead).
STRUCTURED_OUTPUTS_CONFIG='{"disable_any_whitespace":true}'
if [[ -n "${LOCOMO_VLLM_STRUCTURED_OUTPUTS_CONFIG+x}" ]]; then
  STRUCTURED_OUTPUTS_CONFIG="${LOCOMO_VLLM_STRUCTURED_OUTPUTS_CONFIG}"
fi
STRUCTURED_OUTPUTS_ARGS=()
if [[ -n "${STRUCTURED_OUTPUTS_CONFIG}" ]]; then
  STRUCTURED_OUTPUTS_ARGS=(--structured-outputs-config "${STRUCTURED_OUTPUTS_CONFIG}")
fi

exec vllm serve "${MODEL}" \
  "${QUANTIZATION_ARGS[@]}" \
  "${STRUCTURED_OUTPUTS_ARGS[@]}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --api-key "${API_KEY}"
