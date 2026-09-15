from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.memory.throughput.stores import search_store_for_kv


def test_kv_store_search_returns_device_result_without_python_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings: list[str] = []
    query_embedding = np.asarray([0.25, 0.75], dtype=np.float32)
    device_result = object()

    class VectorStore:
        last_search_metrics = SimpleNamespace(search_time_ms=2.5)

        def search_device(self, **kwargs):
            assert kwargs["vectors"] is query_embedding
            return device_result

        def search(self, **_kwargs):
            raise AssertionError(
                "successful device search must skip SearchHit materialization"
            )

    memory = SimpleNamespace(vector_store=VectorStore())
    monkeypatch.setattr(
        "benchmarks.memory.throughput.stores.embed_mem0_query",
        lambda _memory, query: embeddings.append(query) or query_embedding,
    )

    result = search_store_for_kv(
        memory,
        "query",
        top_k=2,
    )

    assert embeddings == ["query"]
    assert result.device_result is device_result
    assert result.search_s == 0.0025


def test_kv_store_search_rejects_unavailable_device_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embed_calls: list[str] = []
    query_embedding = np.asarray([0.25, 0.75], dtype=np.float32)
    search_vectors: list[object] = []

    class VectorStore:
        last_search_metrics = SimpleNamespace(search_time_ms=4.0)

        def search_device(self, **kwargs):
            assert kwargs["vectors"] is query_embedding
            return None

        def search(self, **kwargs):
            search_vectors.append(kwargs["vectors"])
            return ["hit"]

    memory = SimpleNamespace(vector_store=VectorStore())
    monkeypatch.setattr(
        "benchmarks.memory.throughput.stores.embed_mem0_query",
        lambda _memory, query: embed_calls.append(query) or query_embedding,
    )

    with pytest.raises(RuntimeError, match="requires GPU device results"):
        search_store_for_kv(memory, "query", top_k=2)

    assert embed_calls == ["query"]
    assert search_vectors == []


def test_kv_store_search_requires_device_search(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(**_kwargs):
        raise AssertionError("KV selection must never fall back to host results")

    memory = SimpleNamespace(vector_store=SimpleNamespace(search=search))
    monkeypatch.setattr(
        "benchmarks.memory.throughput.stores.embed_mem0_query",
        lambda _memory, _query: [0.25, 0.75],
    )

    with pytest.raises(RuntimeError, match="requires a vector store with GPU device search"):
        search_store_for_kv(memory, "query", top_k=2)
