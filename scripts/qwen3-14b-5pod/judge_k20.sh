#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-${RUN_STAMP:-}}"
: "${STAMP:?Set STAMP or RUN_STAMP to the generation run stamp}"

STAMP="${STAMP}" RUNIDS_FROM="grid" MODELS="qwen3-14b" TOPKS="20" \
  bash "${SCRIPT_DIR}/../judge.sh"
