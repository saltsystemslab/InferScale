#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MODELS="llama mistral qwen"
export TOPKS="10"

exec bash "${SCRIPT_DIR}/../full_run.sh"
