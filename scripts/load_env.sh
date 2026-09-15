#!/usr/bin/env bash
# Source this fixed workflow to load credentials and the default runtime JSON.
if (($#)); then
  echo "load_env.sh takes no arguments; edit configs/runtime.json." >&2
  return 2 2>/dev/null || exit 2
fi
_LOAD_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/environment.sh
source "${_LOAD_ENV_SCRIPT_DIR}/environment.sh"
load_benchmark_environment "${PROJECT_ROOT}/configs/runtime.json" runtime
unset _LOAD_ENV_SCRIPT_DIR
