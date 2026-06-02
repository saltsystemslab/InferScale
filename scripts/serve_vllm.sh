#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
API_KEY="${VLLM_API_KEY:-token-abc123}"
TP="${VLLM_TP:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
DTYPE="${VLLM_DTYPE:-auto}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"

export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${BENCHMARK_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${BENCHMARK_CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BENCHMARK_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${BENCHMARK_CACHE_ROOT}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${BENCHMARK_CACHE_ROOT}/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${BENCHMARK_CACHE_ROOT}/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BENCHMARK_CACHE_ROOT}/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${BENCHMARK_CACHE_ROOT}/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${BENCHMARK_CACHE_ROOT}/vllm_config}"
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
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/tmp}"

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${XDG_CACHE_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_CONFIG_ROOT}" \
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
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --api-key "${API_KEY}"
