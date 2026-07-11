#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${RUN_STAMP:?Set the same RUN_STAMP on all five pods}"

MODELS="qwen3-14b" TOPKS="10" bash "${SCRIPT_DIR}/../full_run.sh"
