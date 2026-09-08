#!/usr/bin/env bash
set -euo pipefail
if (($#)); then
  echo "serve_vllm.sh takes no arguments; edit configs/serve.json and its runtime JSON." >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/environment.sh
source "${SCRIPT_DIR}/environment.sh"
cd "${PROJECT_ROOT}"
load_benchmark_environment "${PROJECT_ROOT}/configs/serve.json"
PYTHON="$(benchmark_python)"
export PATH="${VENV_DIR}/bin:${PATH}"
exec "${PYTHON}" - <<'PY'
from pathlib import Path
from benchmarks.common.serve import serve

raise SystemExit(serve(Path("configs/serve.json")))
PY
