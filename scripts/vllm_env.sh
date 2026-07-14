#!/usr/bin/env bash
# Source this file to prepare the environment for running a vLLM server:
# FlashInfer workspace, CUDA module loading, and a CUDA >= 12.8 sanity check.
# Expects scripts/load_env.sh to have been sourced first.

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
