#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if (($#)); then
  echo "setup_remote.sh takes no arguments; edit configs/setup.json and configs/runtime.json." >&2
  exit 2
fi
# shellcheck source=scripts/environment.sh
source "${SCRIPT_DIR}/environment.sh"
load_benchmark_environment "${PROJECT_ROOT}/configs/setup.json" prepare-launch
_SETUP_EXPORTS="$(python3 - <<'PYSETUP'
from pathlib import Path
from benchmarks.common.setup_config import shell_exports
print(shell_exports(Path("configs/setup.json")))
PYSETUP
)"
eval "${_SETUP_EXPORTS}"
unset _SETUP_EXPORTS
# shellcheck source=scripts/runpod_cuda_cmake.sh
source "${SCRIPT_DIR}/runpod_cuda_cmake.sh"

if [[ "${FRESH_REMOTE_BUILD:-0}" == "1" ]]; then
  rm -rf "${VENV_DIR}" .cache tmp jasperpy/build jasperpy/python/jasper/lib/*.so
  # Recreate runtime-backed paths after removing local cache/build entries.
  load_benchmark_environment "${PROJECT_ROOT}/configs/setup.json" prepare-launch
fi

if [[ "${SKIP_SUBMODULE_INIT:-0}" != "1" ]]; then
  git submodule update --init --recursive jasperpy
fi

echo "Using benchmark runtime root: ${BENCHMARK_RUNTIME_ROOT}"
echo "Using benchmark cache root: ${BENCHMARK_CACHE_ROOT}"
echo "Using benchmark results root: ${BENCHMARK_RESULTS_ROOT}"

if [[ ! -f "${LOCOMO_DATASET_PATH}" ]]; then
  echo "Downloading LoCoMo dataset to ${LOCOMO_DATASET_PATH}"
  mkdir -p "$(dirname -- "${LOCOMO_DATASET_PATH}")"
  curl -fL "${LOCOMO_DATASET_URL}" -o "${LOCOMO_DATASET_PATH}"
else
  echo "Using existing LoCoMo dataset: ${LOCOMO_DATASET_PATH}"
fi

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
  echo "warning: CUDA 12.8 was not detected; RTX/B200 Blackwell needs CUDA >=12.8." >&2
  echo "warning: This script pins cu128 wheels." >&2
fi

PYTHON_VERSION="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION#*.}"
if ((
  PYTHON_MAJOR < 3 ||
  (PYTHON_MAJOR == 3 && PYTHON_MINOR < 10) ||
  PYTHON_MAJOR > 3 ||
  (PYTHON_MAJOR == 3 && PYTHON_MINOR >= 14)
)); then
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
python -m pip install -c "${CONSTRAINTS_FILE}" vllm accelerate hf_transfer --extra-index-url "${PYTORCH_INDEX}"

declare -a JASPER_CMAKE_PLATFORM_ARGS=()
configure_runpod_cuda_cmake_args

cmake -S jasperpy -B jasperpy/build \
  -DJASPER_BUILD_FFI=ON \
  -DJASPER_BUILD_CMD=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES}" \
  "${JASPER_CMAKE_PLATFORM_ARGS[@]}"
cmake --build jasperpy/build --parallel
cmake --install jasperpy/build
python -m pip install -e jasperpy/python

python - <<'PY'
import inferscale
import accelerate
import torch
import transformers
import vllm

print("inferscale:", inferscale.__version__)
print("accelerate:", accelerate.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
PY

# Timed runs consume immutable Mem0 fact catalogs, so extraction is part of
# setup: scripts/extract_facts.sh serves each answer model on a temporary
# local vLLM server and runs the bounded preembed protocol against it.
# Set extract_facts=false in the setup JSON to defer extraction.
if [[ "${SKIP_EXTRACTION:-0}" != "1" ]]; then
  # shellcheck source=scripts/launch.sh
  source "${SCRIPT_DIR}/launch.sh"
  (run_launch "${EXTRACTION_LAUNCH_CONFIG}")
else
  echo "Skipping Mem0 fact extraction (setup JSON extract_facts=false); run scripts/extract_facts.sh before the benchmarks."
fi

echo "Setup complete. Next: run the sweep (bash scripts/full_run.sh)."
