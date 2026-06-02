#!/usr/bin/env bash
# Source this file from the repo root on remote GPU machines to keep
# caches, temp files, and benchmark outputs under scratch storage.

_SCRATCH_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${_SCRATCH_ENV_SCRIPT_DIR}/.." && pwd)}"

_SCRATCH_ENV_USER="${USER:-$(id -un)}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${_SCRATCH_ENV_USER}/benchmark-jasper}"
export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-${SCRATCH_ROOT}/cache}"
export BENCHMARK_RESULTS_ROOT="${BENCHMARK_RESULTS_ROOT:-${SCRATCH_ROOT}/results}"
export MEM0_DIR="${MEM0_DIR:-${BENCHMARK_CACHE_ROOT}/mem0}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BENCHMARK_CACHE_ROOT}/xdg}"

export HF_HOME="${HF_HOME:-${BENCHMARK_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${BENCHMARK_CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BENCHMARK_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${BENCHMARK_CACHE_ROOT}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${BENCHMARK_CACHE_ROOT}/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${BENCHMARK_CACHE_ROOT}/cuda}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${BENCHMARK_CACHE_ROOT}/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${BENCHMARK_CACHE_ROOT}/vllm_config}"

_SCRATCH_ENV_DIRS=(
  "${BENCHMARK_CACHE_ROOT}"
  "${BENCHMARK_RESULTS_ROOT}"
  "${MEM0_DIR}"
  "${TMPDIR}"
  "${PIP_CACHE_DIR}"
  "${XDG_CACHE_HOME}"
  "${HF_HOME}"
  "${HF_HUB_CACHE}"
  "${TRANSFORMERS_CACHE}"
  "${TORCH_HOME}"
  "${TRITON_CACHE_DIR}"
  "${TORCHINDUCTOR_CACHE_DIR}"
  "${TORCH_EXTENSIONS_DIR}"
  "${CUDA_CACHE_PATH}"
  "${VLLM_CACHE_ROOT}"
  "${VLLM_CONFIG_ROOT}"
)

if ! mkdir -p "${_SCRATCH_ENV_DIRS[@]}"; then
  echo "error: could not create scratch directories under ${SCRATCH_ROOT}" >&2
  return 1 2>/dev/null || exit 1
fi

_SCRATCH_ENV_CACHE_LINK="${PROJECT_ROOT}/.cache"
if [[ -L "${_SCRATCH_ENV_CACHE_LINK}" ]]; then
  ln -sfn "${BENCHMARK_CACHE_ROOT}" "${_SCRATCH_ENV_CACHE_LINK}"
elif [[ -e "${_SCRATCH_ENV_CACHE_LINK}" ]]; then
  if rmdir "${_SCRATCH_ENV_CACHE_LINK}" 2>/dev/null; then
    ln -s "${BENCHMARK_CACHE_ROOT}" "${_SCRATCH_ENV_CACHE_LINK}"
  else
    echo "warning: ${_SCRATCH_ENV_CACHE_LINK} exists and is not an empty directory or symlink; leaving it unchanged." >&2
    echo "warning: remove it during a fresh rebuild if you want .cache to point at ${BENCHMARK_CACHE_ROOT}." >&2
  fi
else
  ln -s "${BENCHMARK_CACHE_ROOT}" "${_SCRATCH_ENV_CACHE_LINK}"
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Scratch directories are prepared under ${SCRATCH_ROOT}."
  echo "Run 'source scripts/scratch_env.sh' to export these variables in the current shell."
fi

unset _SCRATCH_ENV_CACHE_LINK
unset _SCRATCH_ENV_DIRS
unset _SCRATCH_ENV_SCRIPT_DIR
unset _SCRATCH_ENV_USER
