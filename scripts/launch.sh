#!/usr/bin/env bash
# Shared implementation, sourced by fixed benchmark workflow scripts.
_LAUNCH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/environment.sh
source "${_LAUNCH_SCRIPT_DIR}/environment.sh"

run_launch() {
  local config_path="$1"
  cd "${PROJECT_ROOT}"
  load_benchmark_environment "${config_path}"
  local interpreter
  interpreter="$(benchmark_python)"
  export PATH="${VENV_DIR}/bin:${PATH}"
  exec "${interpreter}" - "${config_path}" <<'PY'
from pathlib import Path
import sys
from benchmarks.common.launcher import launch

raise SystemExit(launch(Path(sys.argv[1])))
PY
}
