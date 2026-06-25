#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

CUDA_MODULE="${CUDA_MODULE-cuda/12.8}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
JASPER_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES:-native}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-constraints-cu128.txt}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ "${FRESH_REMOTE_BUILD:-0}" == "1" ]]; then
  rm -rf "${VENV_DIR}" .cache tmp jasperpy/build jasperpy/python/jasper/lib/*.so
  # Recreate runtime-backed paths after removing the local .cache entry.
  # shellcheck source=scripts/load_env.sh
  source "${SCRIPT_DIR}/load_env.sh"
fi

if [[ "${SKIP_SUBMODULE_INIT:-0}" != "1" ]]; then
  git submodule update --init --recursive
fi

if [[ "${BENCHMARK_USE_SCRATCH:-1}" != "0" ]]; then
  echo "Using benchmark runtime root: ${BENCHMARK_RUNTIME_ROOT}"
else
  echo "Using project-local cache/results directories."
fi
echo "Using benchmark cache root: ${BENCHMARK_CACHE_ROOT}"
echo "Using benchmark results root: ${BENCHMARK_RESULTS_ROOT}"

if [[ -n "${CUDA_MODULE}" ]]; then
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

if [[ "${CUDA_VERSION}" != 12.8* ]]; then
  echo "warning: CUDA 12.8 was not detected; RTX/B200 Blackwell needs CUDA >=12.8 and this script pins cu128 wheels." >&2
fi

PYTHON_VERSION="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION#*.}"
if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 10) || PYTHON_MAJOR > 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR >= 14) )); then
  echo "error: vLLM 0.19.1 requires Python >=3.10,<3.14; found Python ${PYTHON_VERSION}." >&2
  exit 1
fi

python3 -m venv "${VENV_DIR}"
. "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
  echo "constraints file not found: ${CONSTRAINTS_FILE}" >&2
  exit 1
fi

python -m pip install -c "${CONSTRAINTS_FILE}" torch torchvision torchaudio --index-url "${PYTORCH_INDEX}"
python -m pip install -c "${CONSTRAINTS_FILE}" -e ".[dev,jasper]"
python -m pip install -c "${CONSTRAINTS_FILE}" vllm accelerate --extra-index-url "${PYTORCH_INDEX}"

cmake -S jasperpy -B jasperpy/build \
  -DJASPER_BUILD_FFI=ON \
  -DJASPER_BUILD_CMD=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES}"
cmake --build jasperpy/build --parallel
cmake --install jasperpy/build
python -m pip install -e jasperpy/python

python - <<'PY'
import locomo_jasper_bench
import accelerate
import torch
import transformers
import vllm

print("locomo_jasper_bench:", locomo_jasper_bench.__version__)
print("accelerate:", accelerate.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
PY
