#!/usr/bin/env bash
#
# Extract the mistral fact catalogs on this pod.
# Run extract_llama.sh and extract_qwen.sh on the other two pods to
# parallelize extraction; catalogs land on the shared results volume and are
# disjoint per model, so the runs never conflict.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXTRACTION_MODELS="mistral"

exec bash "${SCRIPT_DIR}/../extract_facts.sh"
