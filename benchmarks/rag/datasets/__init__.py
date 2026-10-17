"""Registry of the evaluable RAG datasets, keyed by the name the run JSON uses."""

from __future__ import annotations

from benchmarks.rag.data_types import RagDatasetSpec
from benchmarks.rag.datasets.multihop_rag import MULTIHOP_RAG_SPEC
from benchmarks.rag.datasets.qasper import QASPER_SPEC

DATASETS: dict[str, RagDatasetSpec] = {spec.name: spec for spec in (MULTIHOP_RAG_SPEC, QASPER_SPEC)}


def get_dataset(name: str) -> RagDatasetSpec:
    key = name.strip().lower()
    spec = DATASETS.get(key)
    if spec is None:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown RAG dataset {name!r}; known datasets: {known}.")
    return spec
