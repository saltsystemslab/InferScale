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
qdrant_local = pytest.importorskip("qdrant_client.local.qdrant_local")


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


@pytest.mark.parametrize("query_copy", [False, True])
def test_failed_copy_is_timed_and_hook_is_restored(monkeypatch: pytest.MonkeyPatch, query_copy: bool) -> None:
    error = ValueError("cannot copy")
    module = qdrant_local if query_copy else local_collection
    other_module = local_collection if query_copy else qdrant_local
    other_original = other_module.deepcopy

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(module, "deepcopy", fail)
    ticks = iter([10, 30])
    monkeypatch.setattr(qdrant_profiling, "perf_counter_ns", lambda: next(ticks))
    with pytest.raises(ValueError) as raised:
        with qdrant_profiling.measure_qdrant_deepcopy() as timing:
            module.deepcopy({"data": "fact"})

    assert raised.value is error
    assert (timing.calls, timing.query_calls) == ((0, 1) if query_copy else (1, 0))
    assert (timing.elapsed_ns, timing.query_elapsed_ns) == ((0, 20) if query_copy else (20, 0))
    assert module.deepcopy is fail
    assert other_module.deepcopy is other_original


def test_nested_and_concurrent_queries_have_independent_collectors() -> None:
    original = local_collection.deepcopy
    query_original = qdrant_local.deepcopy
    with qdrant_profiling.measure_qdrant_deepcopy() as outer:
        local_collection.deepcopy({"data": "outer"})
        qdrant_local.deepcopy({"query": [1.0, 0.0]})
        with qdrant_profiling.measure_qdrant_deepcopy() as inner:
            local_collection.deepcopy({"data": "inner"})
            qdrant_local.deepcopy({"query": [0.0, 1.0]})
        local_collection.deepcopy({"data": "outer again"})
        qdrant_local.deepcopy({"query": [1.0, 0.0]})
    assert (outer.calls, inner.calls) == (2, 1)
    assert (outer.query_calls, inner.query_calls) == (2, 1)

    barrier = Barrier(2)

    def run(calls: int) -> tuple[int, int]:
        with qdrant_profiling.measure_qdrant_deepcopy() as timing:
            barrier.wait(timeout=5)
            qdrant_local.deepcopy({"query": [1.0, 0.0]})
            for _ in range(calls):
                local_collection.deepcopy({"metadata": {"data": "fact"}})
            barrier.wait(timeout=5)
        return timing.calls, timing.query_calls

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, 2)
        second = pool.submit(run, 3)
        assert (first.result(), second.result()) == ((2, 1), (3, 1))
    assert local_collection.deepcopy is original
    assert qdrant_local.deepcopy is query_original


def test_unprofiled_thread_is_not_counted() -> None:
    with qdrant_profiling.measure_qdrant_deepcopy() as timing:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(local_collection.deepcopy, {"data": "other"}).result() == {"data": "other"}
            assert pool.submit(qdrant_local.deepcopy, [1.0, 0.0]).result() == [1.0, 0.0]
        local_collection.deepcopy({"data": "measured"})
        qdrant_local.deepcopy([1.0, 0.0])
    assert timing.calls == 1
    assert timing.query_calls == 1


def test_aggregate_merges_overlapping_intervals_without_counting_gaps() -> None:
    totals = qdrant_profiling.DeepcopyTotals()
    first = qdrant_profiling.DeepcopyTiming(150, 2, [(100, 200), (400, 450)])
    second = qdrant_profiling.DeepcopyTiming(130, 2, [(150, 250), (160, 190)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(totals.add, [second, first]))

    assert totals.calls == 4
    assert totals.time_ms == 280 / 1_000_000
    assert totals.wall_time_ms == 200 / 1_000_000
    assert qdrant_profiling.DeepcopyTotals().wall_time_ms == 0


def test_query_copy_intervals_aggregate_separately_from_payload_intervals() -> None:
    totals = qdrant_profiling.DeepcopyTotals()
    first = qdrant_profiling.DeepcopyTiming(
        150, 2, [(100, 200), (400, 450)],
        query_elapsed_ns=100, query_calls=2, query_intervals_ns=[(50, 100), (410, 460)],
    )
    second = qdrant_profiling.DeepcopyTiming(
        130, 2, [(150, 250), (160, 190)],
        query_elapsed_ns=75, query_calls=2, query_intervals_ns=[(75, 125), (80, 105)],
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(totals.add, [first, second]))

    assert (totals.calls, totals.query_calls) == (4, 4)
    assert totals.time_ms == 280 / 1_000_000
    assert totals.wall_time_ms == 200 / 1_000_000
    assert totals.query_time_ms == 175 / 1_000_000
    assert totals.query_wall_time_ms == 125 / 1_000_000
    assert qdrant_profiling.DeepcopyTotals().query_wall_time_ms == 0


def test_aggregate_rejects_missing_query_intervals_before_adding_any_totals() -> None:
    totals = qdrant_profiling.DeepcopyTotals()
    timing = qdrant_profiling.DeepcopyTiming(10, 1, [(20, 30)], query_elapsed_ns=5, query_calls=1)

    with pytest.raises(ValueError, match="query deepcopy timing requires recorded intervals"):
        totals.add(timing)

    assert (totals.calls, totals.query_calls, totals.elapsed_ns, totals.query_elapsed_ns) == (0, 0, 0, 0)
    assert totals.wall_time_ms == totals.query_wall_time_ms == 0


@pytest.mark.parametrize("query_copy", [False, True])
def test_copy_intervals_include_failed_calls(monkeypatch: pytest.MonkeyPatch, query_copy: bool) -> None:
    module = qdrant_local if query_copy else local_collection
    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("copy failed")

    monkeypatch.setattr(module, "deepcopy", fail)
    ticks = iter([10, 30])
    monkeypatch.setattr(qdrant_profiling, "perf_counter_ns", lambda: next(ticks))
    with pytest.raises(ValueError, match="copy failed"):
        with qdrant_profiling.measure_qdrant_deepcopy(record_intervals=True) as timing:
            module.deepcopy({"data": "fact"})

    assert timing.intervals_ns == ([] if query_copy else [(10, 30)])
    assert timing.query_intervals_ns == ([(10, 30)] if query_copy else [])
    assert (timing.elapsed_ns, timing.query_elapsed_ns) == ((0, 20) if query_copy else (20, 0))
    assert module.deepcopy is fail


def test_store_collection_restores_after_errors_and_rejects_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("QDRANT_PROFILE_DEEPCOPY", "1")
    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    try:
        store.add_many([[1., 0.]], [{"data": "entity"}], ["entity"])

        def failed_query(**kwargs: object) -> None:
            qdrant_local.deepcopy({"query": [1.0, 0.0]})
            local_collection.deepcopy({"data": "entity"})
            raise ValueError("entity search failed")

        monkeypatch.setattr(store._client, "query_points", failed_query)
        with store.collect_deepcopy_timings() as totals:
            with pytest.raises(RuntimeError, match="cannot share"):
                with store.collect_deepcopy_timings():
                    pytest.fail("Overlapping query scopes must not replace the collector")
            # Mem0 catches failed entity futures and continues the request.
            with pytest.raises(ValueError, match="entity search failed"):
                store.search([1., 0.], 1)
        assert totals.calls == 1
        assert totals.time_ms > 0
        assert totals.wall_time_ms == totals.time_ms
        assert totals.query_calls == 1
        assert totals.query_time_ms > 0
        assert totals.query_wall_time_ms == totals.query_time_ms
        assert store._deepcopy_collector is None
        with store.collect_deepcopy_timings() as next_totals:
            assert next_totals.calls == 0
            assert next_totals.query_calls == 0
        with pytest.raises(ValueError, match="outer failure"):
            with store.collect_deepcopy_timings():
                raise ValueError("outer failure")
        assert store._deepcopy_collector is None
    finally:
        store.close()


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
        assert unprofiled.qdrant_query_deepcopy_time_ms is None
        assert unprofiled.qdrant_query_deepcopy_calls is None
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
        assert measured.qdrant_query_deepcopy_calls == 1
        assert 0 < measured.qdrant_query_deepcopy_time_ms <= measured.search_time_ms
        hits[0].payload["metadata"]["tags"].append("mutated")
        repeated, repeated_metrics = store.search([1., 0.], 2, {"user_id": "sample"})
        assert repeated == expected
        assert repeated_metrics.qdrant_deepcopy_calls == 2
        assert repeated_metrics.qdrant_query_deepcopy_calls == 1
        absent, absent_metrics = store.search([1., 0.], 2, {"user_id": "missing"})
        assert absent == []
        assert absent_metrics.qdrant_deepcopy_calls == 0
        assert absent_metrics.qdrant_deepcopy_time_ms == 0
        assert absent_metrics.qdrant_query_deepcopy_calls == 1
        assert absent_metrics.qdrant_query_deepcopy_time_ms > 0
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
        assert metrics.qdrant_query_deepcopy_time_ms == 0
        assert metrics.qdrant_query_deepcopy_calls == 0
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


@pytest.mark.parametrize("query_copy", [False, True])
def test_missing_dependency_hook_fails_clearly_without_half_patching(
    monkeypatch: pytest.MonkeyPatch, query_copy: bool,
) -> None:
    module = qdrant_local if query_copy else local_collection
    other_module = local_collection if query_copy else qdrant_local
    other_original = other_module.deepcopy
    monkeypatch.delattr(module, "deepcopy")
    with pytest.raises(RuntimeError, match="no local deepcopy hook"):
        with qdrant_profiling.measure_qdrant_deepcopy():
            pytest.fail("Unsupported diagnostic hook should fail before the query")
    assert other_module.deepcopy is other_original
    assert qdrant_profiling._patch is None


def test_query_copy_forwards_memo_and_counts_recursion_once(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_original, query_original = local_collection.deepcopy, qdrant_local.deepcopy
    ticks = iter([100, 160])
    monkeypatch.setattr(qdrant_profiling, "perf_counter_ns", lambda: next(ticks))
    original = {"nested": [1, {"text": "query"}]}
    original["self"] = original
    memo = {}

    with qdrant_profiling.measure_qdrant_deepcopy(record_intervals=True) as timing:
        result = qdrant_local.deepcopy(original, memo=memo)
        assert copy.deepcopy is query_original

    assert result["self"] is result
    assert result["nested"] == original["nested"]
    assert result["nested"] is not original["nested"]
    assert memo[id(original)] is result
    assert (timing.calls, timing.elapsed_ns, timing.intervals_ns) == (0, 0, [])
    assert timing.query_calls == 1
    assert timing.query_time_ms == 60 / 1_000_000
    assert timing.query_intervals_ns == [(100, 160)]
    assert local_collection.deepcopy is payload_original
    assert qdrant_local.deepcopy is query_original


@pytest.mark.parametrize("record_intervals", [False, True])
def test_real_qdrant_query_copies_query_once_and_each_returned_payload(record_intervals: bool) -> None:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    try:
        client.create_collection("facts", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT))
        client.upsert("facts", [
            models.PointStruct(id=i, vector=vector, payload={"nested": {"tags": [str(i)]}})
            for i, vector in enumerate([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        ])
        with qdrant_profiling.measure_qdrant_deepcopy(record_intervals=record_intervals) as timing:
            result = client.query_points("facts", query=[1.0, 0.0], limit=2, with_payload=True)

        assert [point.id for point in result.points] == [0, 1]
        assert timing.calls == 2
        assert timing.query_calls == 1
        assert timing.elapsed_ns > 0
        assert timing.query_elapsed_ns > 0
        if record_intervals:
            assert len(timing.intervals_ns) == 2
            assert len(timing.query_intervals_ns) == 1
        else:
            assert timing.intervals_ns is timing.query_intervals_ns is None
        result.points[0].payload["nested"]["tags"].append("mutated")
        assert client.retrieve("facts", [0])[0].payload["nested"]["tags"] == ["0"]
    finally:
        client.close()


def test_changed_second_hook_is_detected_and_first_hook_is_restored() -> None:
    payload_original, query_original = local_collection.deepcopy, qdrant_local.deepcopy
    replacement = lambda value: value
    try:
        with qdrant_profiling.measure_qdrant_deepcopy():
            qdrant_local.deepcopy = replacement
            with pytest.raises(RuntimeError, match="hook changed during profiling"):
                with qdrant_profiling.measure_qdrant_deepcopy():
                    pytest.fail("Changed dependency hook should fail before the query")

        assert local_collection.deepcopy is payload_original
        assert qdrant_local.deepcopy is replacement
        assert qdrant_profiling._patch is None
    finally:
        qdrant_local.deepcopy = query_original


def test_interrupted_second_hook_installation_rolls_back_first_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    from qdrant_client import local

    payload_original, query_original = local_collection.deepcopy, qdrant_local.deepcopy
    error = RuntimeError("cannot install query hook")

    class UnsupportedModule:
        __name__ = "unsupported_qdrant_local"

        @property
        def deepcopy(self):
            return query_original

        @deepcopy.setter
        def deepcopy(self, value):
            raise error

    monkeypatch.setattr(local, "qdrant_local", UnsupportedModule())
    with pytest.raises(RuntimeError) as caught:
        with qdrant_profiling.measure_qdrant_deepcopy():
            pytest.fail("Interrupted installation must not enter the query")

    assert caught.value is error
    assert local_collection.deepcopy is payload_original
    assert qdrant_local.deepcopy is query_original
    assert qdrant_profiling._patch is None
