from __future__ import annotations

import sys

import numpy as np
import pytest

from locomo_jasper_bench.jasper_store import JasperVectorStore, VectorStoreConfig


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
    def __init__(self, indices, distances):
        self.indices = indices
        self.distances = distances
        self.calls = []

    def search(self, query_tensor, *, k, beam_width):
        self.calls.append({"k": k, "beam_width": beam_width})
        return FakeTensor([self.indices[:k]]), FakeTensor([self.distances[:k]])


def test_jasper_search_preserves_backend_order_duplicates_and_uses_payload_cache(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    store = JasperVectorStore(
        tmp_path,
        VectorStoreConfig(backend="jasper", distance="ip"),
    )
    store.add_many(
        [np.array([1, 0, 0], dtype=np.float32) for _ in range(3)],
        [{"memory": f"item-{index}"} for index in range(3)],
        [str(index) for index in range(3)],
    )
    fake_graph = FakeGraph(indices=[2, 0, 2], distances=[-0.9, -0.8, -0.7])
    store._graph = fake_graph
    store._conn.close()

    hits, metrics = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=3)

    assert fake_graph.calls == [{"k": 3, "beam_width": 64}]
    assert metrics.search_time_ms >= 0.0
    assert [hit.id for hit in hits] == ["2", "0", "2"]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert [hit.score for hit in hits] == pytest.approx([0.9, 0.8, 0.7])


def test_jasper_search_uses_requested_top_k_without_beam_cap(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    store = JasperVectorStore(
        tmp_path,
        VectorStoreConfig(backend="jasper", distance="ip", beam_width=64),
    )
    store.add_many(
        [np.array([1, 0, 0], dtype=np.float32) for _ in range(100)],
        [{"memory": f"item-{index}"} for index in range(100)],
        [str(index) for index in range(100)],
    )
    fake_graph = FakeGraph(
        indices=list(range(100)),
        distances=[-(1.0 - index * 0.001) for index in range(100)],
    )
    store._graph = fake_graph

    hits, _ = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=80)
    store.close()

    assert fake_graph.calls == [{"k": 80, "beam_width": 64}]
    assert len(hits) == 80


def test_jasper_search_skips_missing_ordinals_without_exact_fill(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    store = JasperVectorStore(
        tmp_path,
        VectorStoreConfig(backend="jasper", distance="ip"),
    )
    store.add_many(
        [np.array([1, 0, 0], dtype=np.float32) for _ in range(3)],
        [{"memory": f"item-{index}"} for index in range(3)],
        [str(index) for index in range(3)],
    )
    fake_graph = FakeGraph(indices=[0, 2147483647, 1], distances=[-0.9, float("inf"), -0.8])
    store._graph = fake_graph

    hits, _ = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=3)
    store.close()

    assert fake_graph.calls == [{"k": 3, "beam_width": 64}]
    assert [hit.id for hit in hits] == ["0", "1"]
    assert [hit.rank for hit in hits] == [1, 2]
