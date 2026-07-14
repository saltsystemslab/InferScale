#!/usr/bin/env bash
#
# Judge the top-k=100 partition of the qwen3-14b sweep for a given RUN_STAMP.
# Run-ids are regenerated from the sweep grid so each partition script judges
# exactly its own five runs (4 KV windows + 1 prefix baseline).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAMP="${STAMP:-${RUN_STAMP:?Set RUN_STAMP (or STAMP) to the sweep stamp to judge}}"
export RUNIDS_FROM="grid"
export MODELS="qwen3-14b"
export TOPKS="100"

exec bash "${SCRIPT_DIR}/../judge.sh"
