#!/usr/bin/env bash
set -euo pipefail

if (($# > 1)) || [[ "${1:-check}" != "check" ]]; then
  echo "Usage: bash scripts/qdrant.sh [check]" >&2
  echo "Start or stop the Qdrant CPU Pod in the Runpod console." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/environment.sh
source "${SCRIPT_DIR}/environment.sh"
load_benchmark_environment "${PROJECT_ROOT}/configs/runtime.json" runtime
"$(benchmark_python)" -m benchmarks.common.qdrant_check
