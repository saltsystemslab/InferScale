#!/usr/bin/env bash
# Shared implementation, sourced by fixed workflow scripts.
_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${_ENV_SCRIPT_DIR}/.." && pwd)"

load_benchmark_environment() {
  local config_path="$1"
  local config_kind="${2:-launch}"
  local environment_file="${BENCHMARK_ENV_FILE:-${PROJECT_ROOT}/.env}"
  if [[ -f "${environment_file}" ]]; then
    local had_allexport=0 had_nounset=0
    case $- in *a*) had_allexport=1 ;; esac
    case $- in *u*) had_nounset=1; set +u ;; esac
    set -a
    # shellcheck disable=SC1090
    source "${environment_file}"
    if [[ "${had_allexport}" != "1" ]]; then set +a; fi
    if [[ "${had_nounset}" == "1" ]]; then set -u; fi
  fi
  export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  local bootstrap_python=python3
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    bootstrap_python="${PROJECT_ROOT}/.venv/bin/python"
  fi
  local exports
  exports="$("${bootstrap_python}" - "${config_path}" "${config_kind}" <<'PY'
from pathlib import Path
import sys
from benchmarks.common.config import load_runtime_config
from benchmarks.common.environment import runtime_path_for_config, shell_exports

path = Path(sys.argv[1])
runtime_path = path if sys.argv[2] in {"runtime", "prepare-runtime"} else runtime_path_for_config(path)
print(shell_exports(load_runtime_config(runtime_path), prepare=sys.argv[2].startswith("prepare-")))
PY
  )" || return 1
  # shell_exports validates names and quotes every JSON-derived value.
  eval "${exports}"
}

benchmark_python() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    printf '%s\n' "${VENV_DIR}/bin/python"
  else
    printf '%s\n' python3
  fi
}
