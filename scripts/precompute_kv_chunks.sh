#!/usr/bin/env bash
#
# Pre-encode per-fact KV chunks into the chunk cache for every answer model.
#
# For each (model, context window) this runs the in-process chunked-RoPE
# encoder over every sample's fact catalog and saves the chunks under the
# kv-chunks cache, so benchmark runs (accuracy vllm-kv and throughput
# kv_injection) skip the HF model load and the per-fact encode entirely.
# Idempotent: samples whose cache file already exists are skipped.
#
# Fact catalogs must exist first: bash scripts/extract_facts.sh.
# No vLLM server is needed; the encoder runs in-process on the GPU.
#
# Usage:
#   bash scripts/precompute_kv_chunks.sh
#   PRECOMPUTE_MODELS="llama" KV_WINDOWS="50" bash scripts/precompute_kv_chunks.sh
#
# Env knobs:
#   PRECOMPUTE_MODELS  answer models to precompute for (default: llama mistral qwen qwen3-14b)
#   KV_WINDOWS         context windows to precompute (default: 0 5 20 50)
#   DATASET            LoCoMo dataset path (default: data/locomo10.json)
#   MAX_SAMPLES        samples to precompute (default: 10)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/load_env.sh
source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${VIRTUAL_ENV:-}" && -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

PRECOMPUTE_MODELS="${PRECOMPUTE_MODELS:-llama mistral qwen qwen3-14b}"
KV_WINDOWS="${KV_WINDOWS:-0 5 20 50}"
DATASET="${DATASET:-data/locomo10.json}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"

: "${BENCHMARK_RESULTS_ROOT:?Set BENCHMARK_RESULTS_ROOT (normally exported by scripts/load_env.sh)}"

for MODEL_ALIAS in ${PRECOMPUTE_MODELS}; do
  for W in ${KV_WINDOWS}; do
    echo "=== Precomputing KV chunks: model=${MODEL_ALIAS} context_window=${W} ==="
    locomo-jasper-bench \
      --dataset "${DATASET}" \
      --results-dir "${BENCHMARK_RESULTS_ROOT}" \
      --answer-model "${MODEL_ALIAS}" \
      --context-window "${W}" \
      --max-samples "${MAX_SAMPLES}" \
      --precompute-kv-only
  done
done

echo "=== KV chunk precompute complete for: ${PRECOMPUTE_MODELS} (windows: ${KV_WINDOWS}) ==="
