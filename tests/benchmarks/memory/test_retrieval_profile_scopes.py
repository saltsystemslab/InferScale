from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from benchmarks.memory.mem0 import profiling

mem0_main = pytest.importorskip("mem0.memory.main")


def _memory() -> SimpleNamespace:
    return SimpleNamespace(
        vector_store=SimpleNamespace(_retrieval_profile=None),
        _entity_store=SimpleNamespace(_retrieval_profile=None),
    )


def _assert_released(*memories: SimpleNamespace) -> None:
    assert profiling._active.get() is None
    assert profiling._users == 0
    assert not profiling._owners
    assert not profiling._patches
    for memory in memories:
        assert memory.vector_store._retrieval_profile is None
        assert memory._entity_store._retrieval_profile is None


def _install_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mem0_main, "lemmatize_for_bm25", lambda query: query)

    def rank(query: str, *, fail: bool = False) -> str:
        if fail:
            raise ValueError("ranking failed")
        return query

    def pipeline(memory: SimpleNamespace, query: str, *, fail: bool = False) -> str:
        query = mem0_main.lemmatize_for_bm25(query)
        with profiling.profile_search(memory.vector_store):
            with profiling.profile_stage("backend_search"):
                assert query
        return mem0_main.score_and_rank(query, fail=fail)

    monkeypatch.setattr(mem0_main, "score_and_rank", rank)
    monkeypatch.setattr(mem0_main.Memory, "_search_vector_store", pipeline)


@pytest.mark.parametrize(
    "parents,children,expected_ns",
    [
        ([], [(0, 100)], 0),
        ([(0, 10), (8, 20), (30, 40)], [], 30),
        ([(0, 10), (8, 20), (30, 40)], [(-5, 5), (3, 12), (15, 35), (100, 200)], 8),
        ([(10, 20)], [(0, 30), (12, 18)], 0),
        ([(10, 20)], [(0, 10), (20, 30)], 10),
    ],
)
def test_residual_unions_overlaps_and_clips_children_to_parent_intervals(
    parents: list[tuple[int, int]], children: list[tuple[int, int]], expected_ns: int,
) -> None:
    assert profiling.residual_ms(parents, children) == pytest.approx(expected_ns / 1_000_000)


def test_metric_residuals_use_coverage_and_keep_thread_sums_separate() -> None:
    profile = profiling.RetrievalProfile(backend="qdrant")
    milliseconds = {
        "retrieval": [(0, 100)],
        "query_embedding": [(0, 10)],
        "primary_search": [(10, 20)],
        "entity_boost": [(20, 80)],
        "entity_embedding": [(10, 30)],
        "entity_search": [(35, 50), (45, 70), (100, 110)],
        "entity_backend_search": [(36, 48), (46, 65)],
        "entity_payload_access": [(34, 39), (40, 41), (63, 68)],
        "ranking": [(80, 90)],
        "result_format": [(90, 95)],
    }
    profile.add({
        stage: [(start * 1_000_000, stop * 1_000_000) for start, stop in spans]
        for stage, spans in milliseconds.items()
    })

    metrics = profile.metrics()

    assert metrics["mem0_entity_search_time_ms"] == 50
    assert metrics["mem0_entity_search_wall_time_ms"] == 45
    assert metrics["mem0_entity_search_calls"] == 3
    assert metrics["mem0_entity_coordination_residual_time_ms"] == 15
    assert metrics["mem0_entity_backend_residual_wall_time_ms"] == 23
    assert metrics["mem0_request_residual_time_ms"] == 5


def test_concurrent_profiles_and_unprofiled_thread_keep_request_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")
    _install_pipeline(monkeypatch)
    original_lemma = mem0_main.lemmatize_for_bm25
    original_pipeline = mem0_main.Memory._search_vector_store
    memories = [_memory(), _memory(), _memory()]
    barrier = Barrier(3)

    def run_profiled(memory: SimpleNamespace, calls: int) -> profiling.RetrievalProfile:
        with profiling.measure_retrieval(memory, backend="qdrant") as profile:
            barrier.wait(timeout=5)
            with profiling.profile_stage("retrieval"):
                for _ in range(calls):
                    assert mem0_main.Memory._search_vector_store(memory, "query") == "query"
            barrier.wait(timeout=5)
        assert profiling._active.get() is None
        return profile

    def run_unprofiled() -> None:
        barrier.wait(timeout=5)
        for _ in range(5):
            assert mem0_main.Memory._search_vector_store(memories[2], "other") == "other"
        barrier.wait(timeout=5)
        assert profiling._active.get() is None

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(run_profiled, memories[0], 1)
        second = pool.submit(run_profiled, memories[1], 2)
        unprofiled = pool.submit(run_unprofiled)
        profiles = [first.result(timeout=10), second.result(timeout=10)]
        unprofiled.result(timeout=10)

    for calls, profile in enumerate(profiles, start=1):
        assert len(profile.intervals("lemmatization")) == calls
        assert len(profile.intervals("ranking")) == calls
        assert len(profile.intervals("result_format")) == calls
        assert profile.metrics()["mem0_primary_search_calls"] == calls
    assert mem0_main.lemmatize_for_bm25 is original_lemma
    assert mem0_main.Memory._search_vector_store is original_pipeline
    _assert_released(*memories)


@pytest.mark.parametrize("shared", ["memory", "primary", "entity"])
def test_rejected_shared_scope_does_not_replace_or_release_owner_bindings(
    monkeypatch: pytest.MonkeyPatch, shared: str,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")
    owner, rejected = _memory(), _memory()
    if shared == "memory":
        rejected = owner
    elif shared == "primary":
        rejected.vector_store = owner.vector_store
    else:
        rejected._entity_store = owner._entity_store

    with profiling.measure_retrieval(owner, backend="qdrant") as profile:
        bindings = owner.vector_store._retrieval_profile, owner._entity_store._retrieval_profile
        with pytest.raises(RuntimeError, match="cannot share"):
            with profiling.measure_retrieval(rejected, backend="qdrant"):
                pytest.fail("The second scope must not be entered")
        assert owner.vector_store._retrieval_profile is bindings[0]
        assert owner._entity_store._retrieval_profile is bindings[1]
        with profiling.profile_search(owner._entity_store):
            with profiling.profile_stage("backend_search"):
                pass

    assert profile.metrics()["mem0_entity_search_calls"] == 1
    _assert_released(owner, rejected)


def test_non_target_pipeline_and_entity_method_do_not_use_outer_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")
    _install_pipeline(monkeypatch)
    monkeypatch.setattr(
        mem0_main.Memory, "_compute_entity_boosts",
        lambda memory, query: mem0_main.lemmatize_for_bm25(query),
    )
    owner, unrelated = _memory(), _memory()

    with profiling.measure_retrieval(owner, backend="qdrant") as profile:
        assert mem0_main.Memory._search_vector_store(unrelated, "unrelated") == "unrelated"
        assert mem0_main.Memory._compute_entity_boosts(unrelated, "unrelated") == "unrelated"
        assert mem0_main.Memory._search_vector_store(owner, "owner") == "owner"

    assert len(profile.intervals("lemmatization")) == 1
    assert len(profile.intervals("ranking")) == 1
    assert len(profile.intervals("result_format")) == 1
    assert not profile.intervals("entity_boost")
    assert profile.metrics()["mem0_primary_search_calls"] == 1
    _assert_released(owner, unrelated)


def test_one_adapter_cannot_be_attributed_to_both_primary_and_entity_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")
    memory = _memory()
    memory._entity_store = memory.vector_store

    with pytest.raises(RuntimeError, match="distinct primary and entity adapters"):
        with profiling.measure_retrieval(memory, backend="qdrant"):
            pytest.fail("Ambiguous role attribution must prevent scope entry")

    _assert_released(memory)


def test_failed_ranking_has_no_format_tail_and_next_call_starts_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")
    _install_pipeline(monkeypatch)
    memory = _memory()

    with profiling.measure_retrieval(memory, backend="qdrant") as profile:
        for fail in (False, True, False):
            if fail:
                with pytest.raises(ValueError, match="ranking failed"):
                    mem0_main.Memory._search_vector_store(memory, "query", fail=True)
            else:
                assert mem0_main.Memory._search_vector_store(memory, "query") == "query"

    assert len(profile.intervals("ranking")) == 3
    assert len(profile.intervals("result_format")) == 2
    _assert_released(memory)


def test_failed_hook_and_backend_restore_originals_and_keep_failed_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")

    def fail(query: str) -> None:
        raise ValueError("lemmatization failed")

    monkeypatch.setattr(mem0_main, "lemmatize_for_bm25", fail)
    memory = _memory()
    with pytest.raises(ValueError, match="lemmatization failed"):
        with profiling.measure_retrieval(memory, backend="qdrant") as profile:
            originals = [(owner, name, original) for owner, name, original, _ in profiling._patches]
            with pytest.raises(RuntimeError, match="backend failed"):
                with profiling.profile_search(memory._entity_store):
                    with profiling.profile_stage("backend_search"):
                        raise RuntimeError("backend failed")
            mem0_main.lemmatize_for_bm25("query")

    assert len(profile.intervals("lemmatization")) == 1
    assert len(profile.intervals("entity_backend_search")) == 1
    assert profile.metrics()["mem0_entity_search_calls"] == 1
    for owner, name, original in originals:
        assert getattr(owner, name) is original
    _assert_released(memory)


@pytest.mark.parametrize("hook_name", ["score_and_rank", "_search_vector_store", "construct"])
def test_missing_required_hook_fails_before_any_alias_is_replaced(
    monkeypatch: pytest.MonkeyPatch, hook_name: str,
) -> None:
    from qdrant_client.local import local_collection

    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")
    owner = {
        "score_and_rank": mem0_main,
        "_search_vector_store": mem0_main.Memory,
        "construct": local_collection,
    }[hook_name]
    monkeypatch.setattr(owner, hook_name, None)
    original_lemma = mem0_main.lemmatize_for_bm25
    original_entities = mem0_main.extract_entities
    memory = _memory()

    with pytest.raises(RuntimeError, match=f"requires callable {hook_name}"):
        with profiling.measure_retrieval(memory, backend="qdrant"):
            pytest.fail("Invalid hook must prevent scope entry")

    assert mem0_main.lemmatize_for_bm25 is original_lemma
    assert mem0_main.extract_entities is original_entities
    assert getattr(owner, hook_name) is None
    _assert_released(memory)


def test_partial_hook_installation_restores_aliases_and_allows_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")
    original_lemma = mem0_main.lemmatize_for_bm25
    original_entities = mem0_main.extract_entities
    original_ranking = mem0_main.score_and_rank
    memory = _memory()
    attempts = 0

    def fail_once(owner: object, name: str, value: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise RuntimeError("hook installation failed")
        builtins.setattr(owner, name, value)

    monkeypatch.setattr(profiling, "setattr", fail_once, raising=False)
    with pytest.raises(RuntimeError, match="hook installation failed"):
        with profiling.measure_retrieval(memory, backend="qdrant"):
            pytest.fail("Partially installed hooks must not enter the request")

    assert mem0_main.lemmatize_for_bm25 is original_lemma
    assert mem0_main.extract_entities is original_entities
    assert mem0_main.score_and_rank is original_ranking
    _assert_released(memory)
    with profiling.measure_retrieval(memory, backend="qdrant") as profile:
        with profiling.profile_stage("retrieval"):
            pass
    assert len(profile.intervals("retrieval")) == 1
    _assert_released(memory)
