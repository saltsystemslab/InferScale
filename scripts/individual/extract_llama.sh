#!/usr/bin/env bash
#
# Extract the llama fact catalogs on this pod.
# Run extract_qwen.sh and extract_mistral.sh on the other two pods to
# parallelize extraction; catalogs land on the shared results volume and are
# disjoint per model, so the runs never conflict.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXTRACTION_MODELS="llama"

exec bash "${SCRIPT_DIR}/../extract_facts.sh"
