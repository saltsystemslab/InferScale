#!/usr/bin/env bash
set -euo pipefail

CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
VLLM_VERSION="${VLLM_VERSION:-0.10.2}"
JASPER_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES:-native}"

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

python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
cat > "${CONSTRAINTS_FILE}" <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
vllm==${VLLM_VERSION}
EOF

python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX}"

python -m pip install -c "${CONSTRAINTS_FILE}" -e ".[dev,jasper]"
python -m pip install -c "${CONSTRAINTS_FILE}" "vllm==${VLLM_VERSION}" --extra-index-url "${PYTORCH_INDEX}"

cmake -S jasperpy -B jasperpy/build \
  -DJASPER_BUILD_FFI=ON \
  -DJASPER_BUILD_CMD=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES}"
cmake --build jasperpy/build --parallel
cmake --install jasperpy/build
python -m pip install -e jasperpy/python

python - <<'PY'
import locomo_jasper_bench
import torch
import vllm

print("locomo_jasper_bench:", locomo_jasper_bench.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
PY
