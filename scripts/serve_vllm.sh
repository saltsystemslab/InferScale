#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL="${VLLM_MODEL:-shuyuej/Llama-3.3-70B-Instruct-GPTQ}"
API_KEY="${VLLM_API_KEY:-token-abc123}"
TP="${VLLM_TP:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"

export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${BENCHMARK_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${BENCHMARK_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BENCHMARK_CACHE_ROOT}/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${BENCHMARK_CACHE_ROOT}/vllm}"
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
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/tmp}"

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer" \
  "${FLASHINFER_WORKSPACE_DIR}" \
  "${TMPDIR}"

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

if [[ "${CUDA_VERSION}" != 12.9* ]]; then
  echo "warning: CUDA 12.9 runtime was not detected; SM 12.x/Blackwell GPUs may fail." >&2
fi

unset VLLM_MODEL VLLM_API_KEY VLLM_TP VLLM_GPU_MEMORY_UTILIZATION

exec vllm serve "${MODEL}" \
  --quantization gptq \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 4096 \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --api-key "${API_KEY}"
