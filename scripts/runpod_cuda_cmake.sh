#!/usr/bin/env bash

# Populate JASPER_CMAKE_PLATFORM_ARGS for Runpod's target-specific CUDA layout.
configure_runpod_cuda_cmake_args() {
  JASPER_CMAKE_PLATFORM_ARGS=()
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    return 0
  fi

  local nvcc_path
  nvcc_path="$(command -v nvcc || true)"
  if [[ -z "${nvcc_path}" ]]; then
    echo "error: Runpod Jasper builds require nvcc, but it is not on PATH." >&2
    return 1
  fi

  local cuda_root
  cuda_root="$(cd -- "$(dirname -- "$(readlink -f -- "${nvcc_path}")")/.." && pwd -P)"

  local cuda_target
  cuda_target="$(uname -m)-linux"

  local thrust_dir
  thrust_dir="${cuda_root}/targets/${cuda_target}/lib/cmake/thrust"
  if [[ ! -f "${thrust_dir}/ThrustConfig.cmake" && ! -f "${thrust_dir}/thrust-config.cmake" ]]; then
    {
      echo "error: Runpod CUDA installation has no Thrust CMake config at ${thrust_dir}."
      echo "Install the matching CUDA CCCL package, for example:"
      echo "  apt-get update && apt-get install -y cuda-cccl-12-8"
    } >&2
    return 1
  fi

  JASPER_CMAKE_PLATFORM_ARGS=(
    "-DCUDAToolkit_ROOT=${cuda_root}"
    "-DThrust_DIR=${thrust_dir}"
  )
  echo "Runpod CUDA root: ${cuda_root}"
  echo "Runpod Thrust CMake directory: ${thrust_dir}"
}
