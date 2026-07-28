from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from locomo_jasper_bench.throughput.stores import search_store_for_kv


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
        "locomo_jasper_bench.throughput.stores.embed_mem0_query",
        lambda _memory, query: embeddings.append(query) or query_embedding,
    )

    result = search_store_for_kv(
        memory,
        "query",
        top_k=2,
        prefer_device_result=True,
    )

    assert embeddings == ["query"]
    assert result.device_result is device_result
    assert result.hits is None
    assert result.search_s == 0.0025


def test_kv_store_search_reuses_embedding_for_device_unavailable_fallback(
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
        "locomo_jasper_bench.throughput.stores.embed_mem0_query",
        lambda _memory, query: embed_calls.append(query) or query_embedding,
    )

    result = search_store_for_kv(
        memory,
        "query",
        top_k=2,
        prefer_device_result=True,
    )

    assert embed_calls == ["query"]
    assert search_vectors == [query_embedding]
    assert result.device_result is None
    assert result.hits == ["hit"]
    assert result.search_s == 0.004
