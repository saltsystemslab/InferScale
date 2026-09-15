#!/usr/bin/env bash
# Compatibility entry point: edit storage in configs/runtime.json.
if (($#)); then
  echo "scratch_env.sh takes no arguments; edit configs/runtime.json." >&2
  return 2 2>/dev/null || exit 2
fi
_SCRATCH_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/environment.sh
source "${_SCRATCH_ENV_SCRIPT_DIR}/environment.sh"
load_benchmark_environment "${PROJECT_ROOT}/configs/runtime.json" prepare-runtime
unset _SCRATCH_ENV_SCRIPT_DIR
