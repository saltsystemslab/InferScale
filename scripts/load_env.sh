#!/usr/bin/env bash
# Source this file to load local .env values and prepare benchmark paths.

_LOAD_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${_LOAD_ENV_SCRIPT_DIR}/.." && pwd)}"
_LOAD_ENV_FILE="${BENCHMARK_ENV_FILE:-${PROJECT_ROOT}/.env}"

if [[ -f "${_LOAD_ENV_FILE}" ]]; then
  _LOAD_ENV_HAD_NOUNSET=0
  _LOAD_ENV_HAD_ALLEXPORT=0
  case $- in
    *u*)
      _LOAD_ENV_HAD_NOUNSET=1
      set +u
      ;;
  esac
  case $- in
    *a*) _LOAD_ENV_HAD_ALLEXPORT=1 ;;
  esac

  set -a
  # shellcheck disable=SC1090
  source "${_LOAD_ENV_FILE}"
  if [[ "${_LOAD_ENV_HAD_ALLEXPORT}" != "1" ]]; then
    set +a
  fi
  if [[ "${_LOAD_ENV_HAD_NOUNSET}" == "1" ]]; then
    set -u
  fi
fi

unset VLLM_MODEL VLLM_API_KEY VLLM_TP VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN VLLM_DTYPE VLLM_QUANTIZATION VLLM_BASE_URL

if [[ "${BENCHMARK_USE_SCRATCH:-1}" != "0" ]]; then
  # shellcheck source=scripts/scratch_env.sh
  source "${_LOAD_ENV_SCRIPT_DIR}/scratch_env.sh"
else
  export BENCHMARK_RUNTIME_ROOT="${BENCHMARK_RUNTIME_ROOT:-${PROJECT_ROOT}}"
  export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
  export BENCHMARK_RESULTS_ROOT="${BENCHMARK_RESULTS_ROOT:-${PROJECT_ROOT}/results}"
  export MEM0_DIR="${MEM0_DIR:-${BENCHMARK_CACHE_ROOT}/mem0}"
  export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/tmp}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${BENCHMARK_CACHE_ROOT}/pip}"
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

  _LOAD_ENV_DIRS=(
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

  if ! mkdir -p "${_LOAD_ENV_DIRS[@]}"; then
    echo "error: could not create local benchmark directories under ${PROJECT_ROOT}" >&2
    return 1 2>/dev/null || exit 1
  fi
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Environment loaded from ${_LOAD_ENV_FILE}."
  echo "Run 'source scripts/load_env.sh' to export these variables in the current shell."
fi

unset _LOAD_ENV_DIRS
unset _LOAD_ENV_FILE
unset _LOAD_ENV_HAD_ALLEXPORT
unset _LOAD_ENV_HAD_NOUNSET
unset _LOAD_ENV_SCRIPT_DIR
