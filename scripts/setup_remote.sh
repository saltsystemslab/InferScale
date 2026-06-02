#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install -e ".[dev,jasper]"
python -m pip install vllm

JASPER_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES:-native}"

cmake -S jasperpy -B jasperpy/build \
  -DJASPER_BUILD_FFI=ON \
  -DJASPER_BUILD_CMD=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${JASPER_CUDA_ARCHITECTURES}"
cmake --build jasperpy/build --parallel
cmake --install jasperpy/build
python -m pip install -e jasperpy/python

python -c "import locomo_jasper_bench; print(locomo_jasper_bench.__version__)"
