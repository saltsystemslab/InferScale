#!/usr/bin/env bash
#
# Extract the qwen3-14b fact catalogs on this pod.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXTRACTION_MODELS="qwen3-14b"

exec bash "${SCRIPT_DIR}/../extract_facts.sh"
