from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.common.vector_types import SearchHit, SearchMetrics, VectorStoreConfig
from benchmarks.memory.mem0 import adapter as adapter_module
from benchmarks.memory.mem0 import profiling
from benchmarks.memory.mem0.adapter import Mem0JasperVectorStore


def _adapter(store: object, *, backend: str = "qdrant") -> Mem0JasperVectorStore:
    adapter = object.__new__(Mem0JasperVectorStore)
    adapter.config = VectorStoreConfig(backend=backend)
    adapter.store = store
    adapter._retrieval_profile = None
    adapter.last_search_metrics = SearchMetrics(0.0, vector_backend=backend)
    return adapter


def _hit() -> SearchHit:
    return SearchHit(
        id="fact", payload={"linked_memory_ids": ["linked-fact"]},
        score=0.75, distance=-0.75, rank=1,
    )


@pytest.mark.parametrize("backend", ["jasper", "qdrant"])
def test_unprofiled_adapter_preserves_backend_arguments_results_and_metrics(backend: str) -> None:
    hits = [_hit()]
    metrics = SearchMetrics(12.5, vector_backend=backend)
    filters = {"user_id": "sample"}
    calls = []

    def search(vector, *, top_k, filters):
        calls.append((vector, top_k, filters))
        return hits, metrics

    adapter = _adapter(SimpleNamespace(search=search, count=lambda filters: 1), backend=backend)
    returned = adapter.search("question", [[0.25, 0.75]], top_k=5, filters=filters)

    assert returned is hits
    assert adapter.last_search_metrics is metrics
    np.testing.assert_array_equal(calls[0][0], np.array([0.25, 0.75], dtype=np.float32))
    assert calls[0][1] == 5
    assert calls[0][2] is filters
    assert returned[0].payload is hits[0].payload


def test_unprofiled_adapter_preserves_validation_and_previous_metrics_on_failure() -> None:
    hits = [_hit()]
    adapter = _adapter(SimpleNamespace(
        search=lambda vector, **kwargs: (hits, SearchMetrics(9.0)),
        count=lambda filters: 2,
    ))
    previous_metrics = adapter.last_search_metrics

    with pytest.raises(RuntimeError, match="returned 1 hits, expected 2"):
        adapter.search("question", [1.0, 0.0], top_k=2)

    assert adapter.last_search_metrics is previous_metrics


@pytest.mark.parametrize("backend", ["jasper", "qdrant"])
@pytest.mark.parametrize("role", ["primary", "entity"])
def test_profiled_adapter_attributes_stages_without_changing_results(
    monkeypatch: pytest.MonkeyPatch, backend: str, role: str,
) -> None:
    now = 0

    def advance(milliseconds: int) -> None:
        nonlocal now
        now += milliseconds * 1_000_000

    monkeypatch.setattr(profiling, "perf_counter_ns", lambda: now)
    original_first_vector = adapter_module._first_vector
    original_validate = adapter_module._validate_search_hits

    def prepare(vectors):
        advance(5)
        return original_first_vector(vectors)

    def validate(*args, **kwargs):
        advance(23)
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "_first_vector", prepare)
    monkeypatch.setattr(adapter_module, "_validate_search_hits", validate)
    hits = [_hit()]
    metrics = SearchMetrics(12.5, vector_backend=backend)

    def search(vector, **kwargs):
        advance(11)
        with profiling.profile_stage("payload_access"):
            advance(7)
        advance(12)
        return hits, metrics

    def count(filters):
        advance(17)
        return 1

    adapter = _adapter(SimpleNamespace(search=search, count=count), backend=backend)
    profile = profiling.RetrievalProfile(backend=backend)
    adapter._retrieval_profile = (profile, role)

    assert adapter.search("question", [0.25, 0.75], top_k=1) is hits
    assert adapter.last_search_metrics is metrics
    measured = profile.metrics()
    for stage, elapsed in {
        "search": 75, "adapter_prepare": 5, "backend_search": 30,
        "payload_access": 7, "adapter_count": 17, "adapter_validation": 23,
    }.items():
        assert measured[f"mem0_{role}_{stage}_time_ms"] == elapsed
    assert measured[f"mem0_{role}_search_calls"] == 1
    assert profile.intervals(f"{role}_payload_access") == ((16_000_000, 23_000_000),)
    assert profile.intervals(f"{role}_search") == ((0, 75_000_000),)
    with profiling.profile_stage("outside"):
        advance(1)
    assert profile.intervals(f"{role}_outside") == ()


def test_entity_worker_profiles_accumulate_concurrent_searches_and_restore_context() -> None:
    barrier = Barrier(2)
    hits = [_hit()]

    def search(vector, **kwargs):
        with profiling.profile_stage("payload_access"):
            barrier.wait(timeout=5)
        return hits, SearchMetrics(1.0, vector_backend="qdrant")

    adapter = _adapter(SimpleNamespace(search=search, count=lambda filters: 1))
    profile = profiling.RetrievalProfile(backend="qdrant")
    adapter._retrieval_profile = (profile, "entity")

    def run():
        result = adapter.search("entity", [1.0, 0.0], top_k=1)
        with profiling.profile_stage("after_search"):
            pass
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        assert all(future.result(timeout=5) is hits for future in futures)

    assert len(profile.intervals("entity_search")) == 2
    assert len(profile.intervals("entity_payload_access")) == 2
    assert len(profile.intervals("entity_adapter_validation")) == 2
    assert profile.intervals("entity_after_search") == ()
    assert profile.intervals("primary_search") == ()
    measured = profile.metrics()
    assert measured["mem0_entity_search_calls"] == 2
    assert 0 < measured["mem0_entity_payload_access_wall_time_ms"] < measured["mem0_entity_payload_access_time_ms"]


@pytest.mark.parametrize("failure_stage", ["backend_search", "adapter_count", "adapter_validation"])
def test_profiled_adapter_records_failures_without_hiding_exception_or_changing_metrics(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str,
) -> None:
    error = RuntimeError("original failure")
    hits = [_hit()]

    def search(vector, **kwargs):
        if failure_stage == "backend_search":
            raise error
        return hits, SearchMetrics(1.0)

    def count(filters):
        if failure_stage == "adapter_count":
            raise error
        return 1

    if failure_stage == "adapter_validation":
        def fail_validation(*args, **kwargs):
            raise error
        monkeypatch.setattr(adapter_module, "_validate_search_hits", fail_validation)

    adapter = _adapter(SimpleNamespace(search=search, count=count))
    previous_metrics = adapter.last_search_metrics
    profile = profiling.RetrievalProfile(backend="qdrant")
    adapter._retrieval_profile = (profile, "entity")
    with pytest.raises(RuntimeError) as caught:
        adapter.search("entity", [1.0, 0.0], top_k=1)

    assert caught.value is error
    assert adapter.last_search_metrics is previous_metrics
    assert len(profile.intervals(f"entity_{failure_stage}")) == 1
    assert len(profile.intervals("entity_search")) == 1
    with profiling.profile_stage("after_failure"):
        pass
    assert profile.intervals("entity_after_failure") == ()
