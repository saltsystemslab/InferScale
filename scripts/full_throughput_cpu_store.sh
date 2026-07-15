#!/usr/bin/env bash
#
# Throughput sweep with the pinned-host (cpu) KV memory store.
# Runs only the kv_injection condition - the sole condition that uses the KV
# store; the mem0 text baselines never touch it, so they come from
# the standard scripts/full_throughput.sh sweep on the gpu store.
#
# Usage:
#   bash scripts/full_throughput_cpu_store.sh
#   DRY_RUN=1 bash scripts/full_throughput_cpu_store.sh
#   MODELS="llama" USER_COUNTS=10,25 KV_STAGING_SLOTS=8 bash scripts/full_throughput_cpu_store.sh
#
# Extra args are forwarded to locomo-throughput-bench.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CONDITIONS="${CONDITIONS:-kv_injection}"
export RUN_ID="${RUN_ID:-throughput-cpu-$(date -u +%Y%m%dT%H%M%SZ)}"

exec bash "${SCRIPT_DIR}/full_throughput.sh" "$@" \
  --kv-store-backend cpu \
  --kv-staging-slots "${KV_STAGING_SLOTS:-4}"
