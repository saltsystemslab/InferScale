#!/usr/bin/env bash
set -euo pipefail
if (($#)); then
  echo "This script takes no arguments; edit configs/launch/rag-judge.json." >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/launch.sh
source "${SCRIPT_DIR}/../launch.sh"
run_launch "${PROJECT_ROOT}/configs/launch/rag-judge.json"
