#!/usr/bin/env bash
#
# Warm the embedding cache for the RAG benchmarks: every corpus chunk and
# every query, per dataset and answer model (chunk boundaries depend on the
# model's tokenizer). Requires OPENAI_API_KEY. Resumable: cached embeddings
# are skipped.
#
# Datasets come from RAG_DATASETS (default "multihoprag qasper"; legacy
# RAG_DATASET is honored when set). Models default per dataset
# (RAG_MODELS_MULTIHOPRAG=llama, RAG_MODELS_QASPER=qwen; QASPER's
# corpus-wide KV does not fit 250 GB of host RAM with llama).
# PREEMBED_MODELS overrides the model list for every dataset.
#
# Usage:
#   bash scripts/rag/preembed.sh
#   RAG_DATASETS="qasper" PREEMBED_MODELS="llama qwen" bash scripts/rag/preembed.sh

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

models_for_dataset() {
  case "$1" in
    qasper) echo "${RAG_MODELS_QASPER:-qwen}" ;;
    *) echo "${RAG_MODELS_MULTIHOPRAG:-llama}" ;;
  esac
}

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY to populate the embedding cache}"

for DATASET_NAME in ${RAG_DATASETS}; do
  DATASET_MODELS="${PREEMBED_MODELS:-$(models_for_dataset "${DATASET_NAME}")}"
  for MODEL_ALIAS in ${DATASET_MODELS}; do
    echo "=== Preembedding RAG dataset=${DATASET_NAME} model=${MODEL_ALIAS} ==="
    rag-jasper-bench \
      --dataset-name "${DATASET_NAME}" \
      --answer-model "${MODEL_ALIAS}" \
      --preembed-only
  done
done

echo "=== RAG preembed complete for datasets: ${RAG_DATASETS} ==="
