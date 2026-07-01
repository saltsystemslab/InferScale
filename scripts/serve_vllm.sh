#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=scripts/load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

DEFAULT_MODEL_LLAMA="meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL_MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_MODEL_QWEN="Qwen/Qwen2.5-7B-Instruct"

resolve_model_name() {
  local raw="$1"
  local normalized
  normalized="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in
    llama|llama3|llama3.1|llama-3.1|llama-3.1-8b-instruct)
      printf '%s\n' "${MODEL_LLAMA:-${LOCOMO_MODEL_LLAMA:-${DEFAULT_MODEL_LLAMA}}}"
      ;;
    mistral|mistral-7b|mistral-7b-instruct-v0.3)
      printf '%s\n' "${MODEL_MISTRAL:-${LOCOMO_MODEL_MISTRAL:-${DEFAULT_MODEL_MISTRAL}}}"
      ;;
    qwen|qwen2.5|qwen2.5-7b|qwen2.5-7b-instruct)
      printf '%s\n' "${MODEL_QWEN:-${LOCOMO_MODEL_QWEN:-${DEFAULT_MODEL_QWEN}}}"
      ;;
    *)
      printf '%s\n' "${raw}"
      ;;
  esac
}

MODEL="$(resolve_model_name "${JUDGE_MODEL:-${LOCOMO_VLLM_MODEL:-Gemma-2-9B-Instruct}}")"
API_KEY="${JUDGE_API_KEY:-${LOCOMO_VLLM_API_KEY:-token-abc123}}"
TP="${LOCOMO_VLLM_TP:-1}"
GPU_MEMORY_UTILIZATION="${LOCOMO_VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
DTYPE="${LOCOMO_VLLM_DTYPE:-auto}"
QUANTIZATION="${LOCOMO_VLLM_QUANTIZATION:-}"

export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
if [[ -z "${FLASHINFER_WORKSPACE_BASE:-}" ]]; then
  CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-.cache}"
  if [[ "${CACHE_ROOT}" == ".cache" ]]; then
    export FLASHINFER_WORKSPACE_BASE="."
  elif [[ "${CACHE_ROOT}" == */.cache ]]; then
    export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT%/.cache}"
  else
    export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}"
  fi
else
  export FLASHINFER_WORKSPACE_BASE
fi
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer}"

mkdir -p \
  "${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer" \
  "${FLASHINFER_WORKSPACE_DIR}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi

  if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE}"
  else
    echo "warning: CUDA_MODULE=${CUDA_MODULE} is set, but the module command is unavailable." >&2
  fi
fi

if command -v nvcc >/dev/null 2>&1; then
  CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
else
  CUDA_VERSION=""
fi

if [[ -n "${CUDA_VERSION}" ]]; then
  CUDA_MAJOR="${CUDA_VERSION%%.*}"
  CUDA_MINOR="${CUDA_VERSION#*.}"
  CUDA_MINOR="${CUDA_MINOR%%.*}"
else
  CUDA_MAJOR=0
  CUDA_MINOR=0
fi

if (( CUDA_MAJOR < 12 || (CUDA_MAJOR == 12 && CUDA_MINOR < 8) )); then
  echo "warning: CUDA >=12.8 runtime was not detected; RTX/B200 Blackwell GPUs may fail." >&2
fi

QUANTIZATION_ARGS=()
if [[ -n "${QUANTIZATION}" ]]; then
  QUANTIZATION_ARGS=(--quantization "${QUANTIZATION}")
fi

unset VLLM_MODEL VLLM_API_KEY VLLM_TP VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN VLLM_DTYPE VLLM_QUANTIZATION VLLM_BASE_URL

exec vllm serve "${MODEL}" \
  "${QUANTIZATION_ARGS[@]}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --api-key "${API_KEY}"
