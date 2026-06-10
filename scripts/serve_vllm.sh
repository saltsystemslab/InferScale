#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=scripts/load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

MODEL="${VLLM_MODEL:-google/gemma-3-12b-it}"
API_KEY="${VLLM_API_KEY:-token-abc123}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL}}"
TP="${VLLM_TP:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
DTYPE="${VLLM_DTYPE:-auto}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"

export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
if [[ -z "${FLASHINFER_WORKSPACE_BASE:-}" ]]; then
  if [[ "${BENCHMARK_CACHE_ROOT}" == */.cache ]]; then
    export FLASHINFER_WORKSPACE_BASE="${BENCHMARK_CACHE_ROOT%/.cache}"
  else
    export FLASHINFER_WORKSPACE_BASE="${BENCHMARK_CACHE_ROOT}"
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

unset VLLM_MODEL VLLM_API_KEY VLLM_TP VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN VLLM_DTYPE VLLM_QUANTIZATION

exec vllm serve "${MODEL}" \
  "${QUANTIZATION_ARGS[@]}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --api-key "${API_KEY}"
