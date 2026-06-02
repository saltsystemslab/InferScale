from __future__ import annotations

import numpy as np

from locomo_jasper_bench.jasper_store import JasperVectorStore, VectorStoreConfig


def test_numpy_store_persists_and_searches(tmp_path):
    store = JasperVectorStore(
        tmp_path,
        VectorStoreConfig(backend="numpy", distance="ip", normalize=True),
    )
    store.add_many(
        [np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)],
        [{"memory": "alpha"}, {"memory": "beta"}],
        ["a", "b"],
    )
    metrics = store.finalize()
    hits, search_metrics = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)
    store.close()

    assert metrics.backend == "numpy"
    assert metrics.indexed_vector_count == 2
    assert search_metrics.embedding_dim == 3
    assert hits[0].id == "a"
    assert hits[0].payload["memory"] == "alpha"

    reopened = JasperVectorStore(
        tmp_path,
        VectorStoreConfig(backend="numpy", distance="ip", normalize=True),
    )
    reopened_hits, _ = reopened.search(np.array([0, 1, 0], dtype=np.float32), top_k=1)
    reopened.close()
    assert reopened_hits[0].id == "b"
