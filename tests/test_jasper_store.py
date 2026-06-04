from __future__ import annotations

import sys

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


class FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to(self, **_):
        return self

    def __getitem__(self, index):
        return FakeTensor(self._values[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def from_numpy(values):
        return FakeTensor(values)


class FakeGraph:
    def __init__(self):
        self.calls = []

    def search(self, query_tensor, *, k, beam_width):
        self.calls.append({"k": k, "beam_width": beam_width})
        return (
            FakeTensor([list(range(k))]),
            FakeTensor([[1.0 - index * 0.01 for index in range(k)]]),
        )


def test_jasper_search_caps_k_to_beam_width(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    store = JasperVectorStore(
        tmp_path,
        VectorStoreConfig(backend="jasper", distance="ip", normalize=True, beam_width=64),
    )
    store.add_many(
        [np.array([1, 0, 0], dtype=np.float32) for _ in range(100)],
        [{"memory": f"item-{index}"} for index in range(100)],
        [str(index) for index in range(100)],
    )
    fake_graph = FakeGraph()
    store._graph = fake_graph

    hits = store._search_jasper(np.array([1, 0, 0], dtype=np.float32), top_k=80)
    store.close()

    assert fake_graph.calls == [{"k": 64, "beam_width": 64}]
    assert len(hits) == 64
