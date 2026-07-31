#!/usr/bin/env bash
#
# Download the MultiHop-RAG dataset into data/multihoprag and sanity-check it.
#
# The dataset's GitHub raw URLs went away; the HuggingFace resolve URLs below
# are plain-curl-able. Override with MULTIHOP_RAG_CORPUS_URL /
# MULTIHOP_RAG_QUERIES_URL / MULTIHOP_RAG_DATA_DIR.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/load_env.sh
source "${PROJECT_ROOT}/scripts/load_env.sh"

# Matches RagBenchConfig's derived default data_dir (data/<dataset-name>).
DATA_DIR="${MULTIHOP_RAG_DATA_DIR:-data/multihoprag}"
CORPUS_URL="${MULTIHOP_RAG_CORPUS_URL:-https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json}"
QUERIES_URL="${MULTIHOP_RAG_QUERIES_URL:-https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json}"

mkdir -p "${DATA_DIR}"
if [[ ! -f "${DATA_DIR}/corpus.json" ]]; then
  echo "Downloading MultiHop-RAG corpus to ${DATA_DIR}/corpus.json"
  curl -fL "${CORPUS_URL}" -o "${DATA_DIR}/corpus.json"
fi
if [[ ! -f "${DATA_DIR}/MultiHopRAG.json" ]]; then
  echo "Downloading MultiHop-RAG queries to ${DATA_DIR}/MultiHopRAG.json"
  curl -fL "${QUERIES_URL}" -o "${DATA_DIR}/MultiHopRAG.json"
fi

# The loader validates schema, unique urls, evidence resolution, and null
# query invariants; it only needs the standard library.
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

from rag_bench.datasets import get_dataset

spec = get_dataset("multihoprag")
docs, queries = spec.load(Path(sys.argv[1]))
print(f"MultiHop-RAG ok: docs={len(docs)} queries={len(queries)}")
PY
