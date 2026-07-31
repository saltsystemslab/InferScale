#!/usr/bin/env bash
#
# Download the RAG benchmark datasets into data/<dataset> and sanity-check
# each with its loader. Datasets come from RAG_DATASETS (default
# "multihoprag qasper"); the legacy single-dataset RAG_DATASET is honored
# when RAG_DATASETS is unset.
#
# MultiHop-RAG: the dataset's GitHub raw URLs went away; the HuggingFace
# resolve URLs below are plain-curl-able. Override with
# MULTIHOP_RAG_CORPUS_URL / MULTIHOP_RAG_QUERIES_URL / MULTIHOP_RAG_DATA_DIR.
# QASPER: the official AllenAI test archive. Override with
# QASPER_ARCHIVE_URL / QASPER_DATA_DIR.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/load_env.sh
source "${PROJECT_ROOT}/scripts/load_env.sh"

RAG_DATASETS="${RAG_DATASETS:-${RAG_DATASET:-multihoprag qasper}}"

for DATASET_NAME in ${RAG_DATASETS}; do
  case "${DATASET_NAME}" in
    multihoprag)
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
      ;;
    qasper)
      DATA_DIR="${QASPER_DATA_DIR:-data/qasper}"
      ARCHIVE_URL="${QASPER_ARCHIVE_URL:-https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz}"
      ARCHIVE_PATH="${DATA_DIR}/qasper-test-and-evaluator-v0.3.tgz"

      mkdir -p "${DATA_DIR}"
      if [[ ! -f "${DATA_DIR}/qasper-test-v0.3.json" ]]; then
        echo "Downloading QASPER test archive to ${ARCHIVE_PATH}"
        curl -fL "${ARCHIVE_URL}" -o "${ARCHIVE_PATH}"
        tar -xzf "${ARCHIVE_PATH}" -C "${DATA_DIR}"
        rm -f "${ARCHIVE_PATH}"
        if [[ ! -f "${DATA_DIR}/qasper-test-v0.3.json" ]]; then
          echo "ERROR: ${ARCHIVE_URL} did not contain qasper-test-v0.3.json" >&2
          exit 1
        fi
      fi
      ;;
    *)
      echo "Unknown dataset '${DATASET_NAME}' in RAG_DATASETS (expected multihoprag or qasper)." >&2
      exit 2
      ;;
  esac

  # The loader validates the schema end to end; it only needs the standard library.
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${DATASET_NAME}" "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

from rag_bench.datasets import get_dataset

spec = get_dataset(sys.argv[1])
docs, queries = spec.load(Path(sys.argv[2]))
print(f"{spec.name} ok: docs={len(docs)} queries={len(queries)}")
PY
done

echo "=== RAG dataset setup complete for: ${RAG_DATASETS} ==="
