#!/usr/bin/env bash
#
# Warm the embedding cache for the RAG benchmark: every corpus chunk and every
# query, per answer model (chunk boundaries depend on the model's tokenizer).
# Requires OPENAI_API_KEY. Resumable: cached embeddings are skipped.
#
# Usage:
#   bash scripts/rag/preembed.sh
#   PREEMBED_MODELS="llama qwen" bash scripts/rag/preembed.sh

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

PREEMBED_MODELS="${PREEMBED_MODELS:-llama}"
DATASET_NAME="${RAG_DATASET:-multihoprag}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY to populate the embedding cache}"

for MODEL_ALIAS in ${PREEMBED_MODELS}; do
  echo "=== Preembedding RAG dataset=${DATASET_NAME} model=${MODEL_ALIAS} ==="
  rag-jasper-bench \
    --dataset-name "${DATASET_NAME}" \
    --answer-model "${MODEL_ALIAS}" \
    --preembed-only
done

echo "=== RAG preembed complete for: ${PREEMBED_MODELS} ==="
