from __future__ import annotations

import numpy as np

from locomo_jasper_bench.config import parse_args
from locomo_jasper_bench.mem0_adapter import Mem0JasperVectorStore
from locomo_jasper_bench.memory_builder import _store_config
from locomo_jasper_bench.vector_types import SearchMetrics


def test_vector_normalize_cli_propagates_to_store_config() -> None:
    config = parse_args(["--vector-normalize"])

    assert config.vector_normalize is True
    assert _store_config(config).normalize_vectors is True


def test_mem0_adapter_l2_normalizes_insert_search_and_exact_search(tmp_path) -> None:
    store = Mem0JasperVectorStore(path=str(tmp_path), normalize_vectors=True)
    capture = _CaptureStore()
    store.store = capture

    store.insert([[3.0, 4.0], [0.0, 0.0]], [{"memory": "one"}, {"memory": "zero"}], ["one", "zero"])
    store.search(query="q", vectors=[10.0, 0.0], top_k=1)
    store.exact_search(query="q", vectors=[[0.0, 5.0]], top_k=1)

    np.testing.assert_allclose(capture.inserted_vectors[0], np.array([0.6, 0.8], dtype=np.float32))
    np.testing.assert_allclose(capture.inserted_vectors[1], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(capture.search_query_vector, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(capture.exact_query_vector, np.array([0.0, 1.0], dtype=np.float32))


def test_mem0_adapter_leaves_vectors_unchanged_when_normalization_disabled(tmp_path) -> None:
    store = Mem0JasperVectorStore(path=str(tmp_path), normalize_vectors=False)
    capture = _CaptureStore()
    store.store = capture

    store.insert([[3.0, 4.0]], [{"memory": "one"}], ["one"])
    store.search(query="q", vectors=[10.0, 0.0], top_k=1)

    np.testing.assert_allclose(capture.inserted_vectors[0], np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(capture.search_query_vector, np.array([10.0, 0.0], dtype=np.float32))


class _CaptureStore:
    vector_count = 0
    dim = None

    def __init__(self) -> None:
        self.inserted_vectors: list[np.ndarray] = []
        self.search_query_vector: np.ndarray | None = None
        self.exact_query_vector: np.ndarray | None = None

    def add_many(
        self,
        vectors: list[object],
        payloads: list[dict[str, object]],
        ids: list[str] | None,
    ) -> list[str]:
        self.inserted_vectors = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        return list(ids or [])

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[object], SearchMetrics]:
        self.search_query_vector = np.asarray(query_vector, dtype=np.float32)
        return [], SearchMetrics(search_time_ms=1.0)

    def exact_search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[object], SearchMetrics]:
        self.exact_query_vector = np.asarray(query_vector, dtype=np.float32)
        return [], SearchMetrics(search_time_ms=1.0)
