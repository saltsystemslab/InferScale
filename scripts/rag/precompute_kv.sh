#!/usr/bin/env bash
#
# Pre-encode every corpus chunk's KV (with its context-window prefix) into the
# RAG per-chunk disk cache, per (dataset, model, window, chunk size).
# Idempotent and resumable: existing chunk files are skipped.
#
# Datasets come from RAG_DATASETS (default "multihoprag qasper"; legacy
# RAG_DATASET is honored when set). Models default per dataset
# (RAG_MODELS_MULTIHOPRAG=llama, RAG_MODELS_QASPER=qwen); PRECOMPUTE_MODELS
# overrides the model list for every dataset.
#
# Before each configuration this prints the --estimate-only projection and
# refuses to start when the cache filesystem's free space (plus what is
# already cached) is below it. Skip the check with RAG_SKIP_DF_CHECK=1.
#
# Usage:
#   bash scripts/rag/precompute_kv.sh
#   RAG_DATASETS="multihoprag" PRECOMPUTE_MODELS="llama" RAG_WINDOWS="5" bash scripts/rag/precompute_kv.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/load_env.sh
source "${PROJECT_ROOT}/scripts/load_env.sh"

if [[ -z "${VIRTUAL_ENV:-}" && -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

RAG_DATASETS="${RAG_DATASETS:-${RAG_DATASET:-multihoprag qasper}}"
RAG_WINDOWS="${RAG_WINDOWS:-5}"
RAG_CHUNK_SIZES="${RAG_CHUNK_SIZES:-1024}"
RAG_SKIP_DF_CHECK="${RAG_SKIP_DF_CHECK:-0}"

models_for_dataset() {
  case "$1" in
    qasper) echo "${RAG_MODELS_QASPER:-qwen}" ;;
    *) echo "${RAG_MODELS_MULTIHOPRAG:-llama}" ;;
  esac
}

CACHE_ROOT="${RAG_KV_CHUNK_CACHE_ROOT:-${BENCHMARK_CACHE_ROOT}/rag-kv-chunks}"
mkdir -p "${CACHE_ROOT}"

for DATASET_NAME in ${RAG_DATASETS}; do
  DATASET_MODELS="${PRECOMPUTE_MODELS:-$(models_for_dataset "${DATASET_NAME}")}"
  for MODEL_ALIAS in ${DATASET_MODELS}; do
    for W in ${RAG_WINDOWS}; do
      for C in ${RAG_CHUNK_SIZES}; do
        echo "=== Estimating RAG KV cache: dataset=${DATASET_NAME} model=${MODEL_ALIAS} window=${W} chunk_size=${C} ==="
        EST_OUT="$(rag-jasper-bench \
          --dataset-name "${DATASET_NAME}" \
          --answer-model "${MODEL_ALIAS}" \
          --chunk-size "${C}" \
          --context-window "${W}" \
          --estimate-only)"
        echo "${EST_OUT}"
        NEEDED_BYTES="$(sed -n 's/.*projected_kv_cache_bytes=\([0-9][0-9]*\).*/\1/p' <<<"${EST_OUT}" | tail -1)"
        NEEDED_BYTES="${NEEDED_BYTES:-0}"
        if [[ "${RAG_SKIP_DF_CHECK}" != "1" && "${NEEDED_BYTES}" -gt 0 ]]; then
          AVAIL_BYTES="$(( $(df -Pk "${CACHE_ROOT}" | awk 'NR==2 {print $4}') * 1024 ))"
          CACHED_BYTES="$(( $(du -sk "${CACHE_ROOT}" 2>/dev/null | awk '{print $1}') * 1024 ))"
          if (( AVAIL_BYTES + CACHED_BYTES < NEEDED_BYTES )); then
            echo "ERROR: projected KV cache needs ${NEEDED_BYTES} bytes but ${CACHE_ROOT} has" >&2
            echo "only ${AVAIL_BYTES} free (${CACHED_BYTES} already cached)." >&2
            echo "Free up space, point RAG_KV_CHUNK_CACHE_ROOT elsewhere, or set RAG_SKIP_DF_CHECK=1." >&2
            exit 1
          fi
        fi
        echo "=== Precomputing RAG KV chunks: dataset=${DATASET_NAME} model=${MODEL_ALIAS} window=${W} chunk_size=${C} ==="
        rag-jasper-bench \
          --dataset-name "${DATASET_NAME}" \
          --answer-model "${MODEL_ALIAS}" \
          --chunk-size "${C}" \
          --context-window "${W}" \
          --precompute-kv-only
      done
    done
  done
done

echo "=== RAG KV precompute complete for datasets: ${RAG_DATASETS} (windows: ${RAG_WINDOWS}, chunk sizes: ${RAG_CHUNK_SIZES}) ==="
