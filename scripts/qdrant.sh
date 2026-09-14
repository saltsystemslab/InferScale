#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if (($# > 1)); then
  echo "Usage: bash scripts/qdrant.sh [start|stop]" >&2
  exit 2
fi

case "${1:-start}" in
  start)
    if ! command -v python3 >/dev/null 2>&1; then
      echo "python3 is required to check Qdrant readiness." >&2
      exit 1
    fi
    docker compose -f "${PROJECT_ROOT}/compose.yaml" up -d qdrant
    python3 - <<'PY'
import sys
import time
import urllib.error
import urllib.request

url = "http://127.0.0.1:6333/readyz"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + 60
last_error = "no response"
while time.monotonic() < deadline:
    try:
        with opener.open(url, timeout=2) as response:
            if response.status == 200:
                print("Qdrant is ready at http://127.0.0.1:6333 (gRPC port 6334).")
                sys.exit(0)
            last_error = f"HTTP {response.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        last_error = str(error)
    time.sleep(1)

print(f"Qdrant did not become ready within 60 seconds: {last_error}", file=sys.stderr)
print("Inspect the server with: docker compose logs qdrant", file=sys.stderr)
sys.exit(1)
PY
    ;;
  stop)
    docker compose -f "${PROJECT_ROOT}/compose.yaml" stop qdrant
    ;;
  *)
    echo "Usage: bash scripts/qdrant.sh [start|stop]" >&2
    exit 2
    ;;
esac
