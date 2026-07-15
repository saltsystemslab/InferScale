#!/usr/bin/env bash
#
# Extract the qwen fact catalogs on this pod.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXTRACTION_MODELS="qwen"

exec bash "${SCRIPT_DIR}/../extract_facts.sh"
