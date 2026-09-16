from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from benchmarks.common.vector_types import SearchMetrics, VectorStoreConfig
from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.mem0.prepared_retriever import PreparedMem0Retriever
from benchmarks.memory.qdrant_index import QdrantVectorStore

Memory = pytest.importorskip("mem0.memory.main").Memory


@pytest.mark.parametrize("profiling", [False, True])
def test_prepared_retrieval_collects_all_mem0_entity_workers_and_resets_per_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profiling: bool,
) -> None:
    monkeypatch.setenv("QDRANT_PROFILE_DEEPCOPY", "1" if profiling else "0")
    config = VectorStoreConfig(backend="qdrant")
    primary = QdrantVectorStore(tmp_path / "primary", config)
    entities = QdrantVectorStore(tmp_path / "entities", config)
    barrier = Barrier(2)
    fact = MemoryFact(
        id="fact-1", text="Alice lives in Paris.", created_at="2023-01-01T00:00:00+00:00",
        timestamp_epoch=1672531200, sample_id="sample", source_session_index=1,
        source_session_id="session_1", source_turn_index=0,
        source_turn_id="sample:session_1:0", speaker="Alice", role="user",
    )

    class EntityAdapter:
        store = entities
        last_search_metrics = SearchMetrics(0.0)

        def search(self, *, query, vectors, top_k, filters):
            # Exercise concurrent worker calls on the same concrete Qdrant store.
            barrier.wait(timeout=5)
            hits, metrics = self.store.search(vectors, top_k, filters)
            self.last_search_metrics = metrics
            return hits

    class FakeMemory:
        _normalize_entity_text = staticmethod(Memory._normalize_entity_text)

        def __init__(self) -> None:
            self.vector_store = SimpleNamespace(last_search_metrics=SearchMetrics(0.0))
            self._entity_store = EntityAdapter()
            self.entity_store = self._entity_store
            self.embedding_model = SimpleNamespace(
                embed_batch=lambda texts, action: [[1., 0.] for _ in texts],
            )

        def search(self, query, *, top_k, filters):
            _, metrics = primary.search([1., 0.], top_k, filters)
            self.vector_store.last_search_metrics = metrics
            if query == "with entities":
                # Use Mem0's actual embedding/ThreadPoolExecutor/boost implementation.
                boosts = Memory._compute_entity_boosts(
                    self, [("PERSON", "Alice"), ("GPE", "Paris")], filters,
                )
                assert boosts[fact.id] > 0
            return {"results": [{
                "memory": fact.text, "score": 0.9, "metadata": {"fact_id": fact.id},
            }]}

    try:
        primary.add_many([[1., 0.]], [{"data": fact.text, "user_id": "sample"}], [fact.id])
        entities.add_many(
            [[1., 0.], [0.9, 0.1], [0.8, 0.2]],
            [{"user_id": "sample", "linked_memory_ids": [fact.id]} for _ in range(3)],
            ["alice", "paris", "another"],
        )
        memory = FakeMemory()
        retriever = PreparedMem0Retriever(
            memory, sample_id="sample", fact_catalog=(fact,), vector_backend="qdrant",
        )
        for query, entity_calls in [("with entities", 6), ("without entities", 0), ("with entities", 6)]:
            hits, metrics = retriever.search(query, top_k=1)
            assert [(hit.id, hit.payload["data"], hit.score) for hit in hits] == [(fact.id, fact.text, 0.9)]
            assert metrics.qdrant_deepcopy_calls == (1 if profiling else None)
            assert metrics.qdrant_entity_deepcopy_calls == (entity_calls if profiling else None)
            assert metrics.qdrant_query_deepcopy_calls == (1 if profiling else None)
            assert metrics.qdrant_entity_query_deepcopy_calls == (entity_calls // 3 if profiling else None)
            if profiling:
                assert isfinite(metrics.qdrant_query_deepcopy_time_ms)
                assert 0 < metrics.qdrant_query_deepcopy_time_ms <= metrics.total_time_ms
                assert 0 <= metrics.qdrant_entity_deepcopy_wall_time_ms <= metrics.qdrant_entity_deepcopy_time_ms
                assert metrics.qdrant_entity_deepcopy_wall_time_ms <= metrics.total_time_ms
                assert isfinite(metrics.qdrant_entity_query_deepcopy_time_ms)
                assert isfinite(metrics.qdrant_entity_query_deepcopy_wall_time_ms)
                assert 0 <= metrics.qdrant_entity_query_deepcopy_wall_time_ms <= metrics.qdrant_entity_query_deepcopy_time_ms
                assert metrics.qdrant_entity_query_deepcopy_wall_time_ms <= metrics.total_time_ms
                if entity_calls:
                    assert metrics.qdrant_entity_deepcopy_time_ms > 0
                    assert metrics.qdrant_entity_query_deepcopy_time_ms > 0
                    # A shared last-search slot contains only one worker's three copies.
                    assert memory._entity_store.last_search_metrics.qdrant_deepcopy_calls == 3
                    assert memory._entity_store.last_search_metrics.qdrant_query_deepcopy_calls == 1
                else:
                    assert metrics.qdrant_entity_deepcopy_time_ms == 0
                    assert metrics.qdrant_entity_deepcopy_wall_time_ms == 0
                    assert metrics.qdrant_entity_query_deepcopy_time_ms == 0
                    assert metrics.qdrant_entity_query_deepcopy_wall_time_ms == 0
            else:
                assert metrics.qdrant_entity_deepcopy_time_ms is None
                assert metrics.qdrant_entity_deepcopy_wall_time_ms is None
                assert metrics.qdrant_query_deepcopy_time_ms is None
                assert metrics.qdrant_entity_query_deepcopy_time_ms is None
                assert metrics.qdrant_entity_query_deepcopy_wall_time_ms is None
            assert entities._deepcopy_collector is None
    finally:
        primary.close()
        entities.close()


def test_rejected_overlapping_query_does_not_restore_another_queries_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("QDRANT_PROFILE_DEEPCOPY", "1")
    entities = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    entered, release, finished = Event(), Event(), Event()
    original_embedder = SimpleNamespace()

    def search(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return {"results": []}

    memory = SimpleNamespace(
        embedding_model=original_embedder, search=search,
        vector_store=SimpleNamespace(last_search_metrics=SearchMetrics(0.0)),
        _entity_store=SimpleNamespace(store=entities),
    )
    retriever = PreparedMem0Retriever(memory, sample_id="sample", fact_catalog=(), vector_backend="qdrant")
    collect = entities.collect_deepcopy_timings

    @contextmanager
    def race_on_rejected_entry():
        try:
            with collect() as totals:
                yield totals
        except RuntimeError:
            # Let the owner restore its embedder before the rejected caller unwinds.
            release.set()
            assert finished.wait(timeout=5)
            raise

    monkeypatch.setattr(entities, "collect_deepcopy_timings", race_on_rejected_entry)

    def first_query():
        try:
            return retriever.search("first", top_k=1)
        finally:
            finished.set()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(first_query)
            assert entered.wait(timeout=5)
            with pytest.raises(RuntimeError, match="cannot share"):
                retriever.search("second", top_k=1)
            first.result(timeout=5)
        assert memory.embedding_model is original_embedder
        assert entities._deepcopy_collector is None
    finally:
        release.set()
        entities.close()
