from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from benchmarks.common.vector_types import VectorStoreConfig
from benchmarks.memory import qdrant_profiling
from benchmarks.memory.qdrant_index import QdrantVectorStore

local_collection = pytest.importorskip("qdrant_client.local.local_collection")


def test_copy_timing_is_inclusive_once_and_restores_original(monkeypatch: pytest.MonkeyPatch) -> None:
    original = local_collection.deepcopy
    ticks = iter([100, 160, 200, 290])
    monkeypatch.setattr(qdrant_profiling, "perf_counter_ns", lambda: next(ticks))
    payload = {"metadata": {"nested": [1, {"text": "fact"}]}}
    payload["self"] = payload

    with qdrant_profiling.measure_qdrant_deepcopy() as timing:
        result = local_collection.deepcopy(payload)
        assert result is not payload
        assert result["self"] is result
        assert result["metadata"] == payload["metadata"]
        local_collection.deepcopy({"data": "another fact"})
        assert copy.deepcopy is original

    assert timing.calls == 2
    assert timing.elapsed_ns == 150
    assert timing.time_ms == 0.00015
    assert local_collection.deepcopy is original


def test_failed_copy_is_timed_and_hook_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ValueError("cannot copy")

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(local_collection, "deepcopy", fail)
    ticks = iter([10, 30])
    monkeypatch.setattr(qdrant_profiling, "perf_counter_ns", lambda: next(ticks))
    with pytest.raises(ValueError) as raised:
        with qdrant_profiling.measure_qdrant_deepcopy() as timing:
            local_collection.deepcopy({"data": "fact"})

    assert raised.value is error
    assert timing.calls == 1
    assert timing.elapsed_ns == 20
    assert local_collection.deepcopy is fail


def test_nested_and_concurrent_queries_have_independent_collectors() -> None:
    original = local_collection.deepcopy
    with qdrant_profiling.measure_qdrant_deepcopy() as outer:
        local_collection.deepcopy({"data": "outer"})
        with qdrant_profiling.measure_qdrant_deepcopy() as inner:
            local_collection.deepcopy({"data": "inner"})
        local_collection.deepcopy({"data": "outer again"})
    assert (outer.calls, inner.calls) == (2, 1)

    barrier = Barrier(2)

    def run(calls: int) -> int:
        with qdrant_profiling.measure_qdrant_deepcopy() as timing:
            barrier.wait(timeout=5)
            for _ in range(calls):
                local_collection.deepcopy({"metadata": {"data": "fact"}})
            barrier.wait(timeout=5)
        return timing.calls

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, 2)
        second = pool.submit(run, 3)
        assert (first.result(), second.result()) == (2, 3)
    assert local_collection.deepcopy is original


def test_unprofiled_thread_is_not_counted() -> None:
    with qdrant_profiling.measure_qdrant_deepcopy() as timing:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(local_collection.deepcopy, {"data": "other"}).result() == {"data": "other"}
        local_collection.deepcopy({"data": "measured"})
    assert timing.calls == 1


def test_profiled_query_preserves_search_results_and_copy_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("QDRANT_PROFILE_DEEPCOPY", "1")
    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    payloads = [
        {"data": "first", "user_id": "sample", "metadata": {"tags": ["a"]}},
        {"data": "second", "user_id": "sample", "metadata": {"tags": ["b"]}},
        {"data": "other", "user_id": "other"},
    ]
    try:
        store.add_many([[1., 0.], [0.5, 0.5], [0., 1.]], payloads, ["first", "second", "other"])
        store.finalize()
        monkeypatch.setattr(store, "_profile_deepcopy", False)
        expected, unprofiled = store.search([1., 0.], 2, {"user_id": "sample"})
        assert unprofiled.qdrant_deepcopy_time_ms is None
        assert unprofiled.qdrant_deepcopy_calls is None
        monkeypatch.setattr(store, "_profile_deepcopy", True)
        query_calls = []
        query_points = store._client.query_points

        def query(**kwargs: object) -> object:
            query_calls.append(kwargs)
            return query_points(**kwargs)

        def unexpected_retrieve(*args: object, **kwargs: object) -> None:
            pytest.fail("Profiling must not add a second payload retrieval call")

        monkeypatch.setattr(store._client, "query_points", query)
        monkeypatch.setattr(store._client, "retrieve", unexpected_retrieve)
        hits, measured = store.search([1., 0.], 2, {"user_id": "sample"})
        assert hits == expected
        assert len(query_calls) == 1
        assert query_calls[0]["with_payload"] is True
        assert measured.qdrant_deepcopy_calls == 2
        assert 0 <= measured.qdrant_deepcopy_time_ms <= measured.search_time_ms
        hits[0].payload["metadata"]["tags"].append("mutated")
        repeated, repeated_metrics = store.search([1., 0.], 2, {"user_id": "sample"})
        assert repeated == expected
        assert repeated_metrics.qdrant_deepcopy_calls == 2
    finally:
        store.close()


def test_empty_profiled_store_reports_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QDRANT_PROFILE_DEEPCOPY", "1")
    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    try:
        hits, metrics = store.search([1., 0.], 2)
        assert hits == []
        assert metrics.qdrant_deepcopy_time_ms == 0
        assert metrics.qdrant_deepcopy_calls == 0
    finally:
        store.close()


def test_diagnostics_are_opt_in_and_validate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_PROFILE_DEEPCOPY", raising=False)
    assert not qdrant_profiling.deepcopy_profiling_enabled()
    monkeypatch.setenv("QDRANT_PROFILE_DEEPCOPY", "1")
    assert qdrant_profiling.deepcopy_profiling_enabled()
    monkeypatch.setenv("QDRANT_PROFILE_DEEPCOPY", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        qdrant_profiling.deepcopy_profiling_enabled()


def test_missing_dependency_hook_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(local_collection, "deepcopy")
    with pytest.raises(RuntimeError, match="no local deepcopy hook"):
        with qdrant_profiling.measure_qdrant_deepcopy():
            pytest.fail("Unsupported diagnostic hook should fail before the query")
