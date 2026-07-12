#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODELS="qwen3-14b"
export TOPKS="5"

exec bash "${SCRIPT_DIR}/../full_run.sh"
