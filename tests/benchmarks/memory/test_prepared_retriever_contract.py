from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.common.vector_types import SearchMetrics
from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.mem0.memory_handles import BuiltMemory
from benchmarks.memory.mem0.prepared_retriever import PreparedMem0Retriever


def _fact(fact_id: str) -> MemoryFact:
    return MemoryFact(
        id=fact_id,
        text=f"Fact {fact_id}",
        created_at="2023-01-01T00:00:00+00:00",
        timestamp_epoch=1672531200,
        sample_id="sample-1",
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=0,
        source_turn_id="sample-1:session_1:0",
        speaker="Alice",
        role="user",
    )


def _built(memory: object, facts: tuple[MemoryFact, ...]) -> BuiltMemory:
    return BuiltMemory(memory=memory, facts=facts, metrics={}, embedder=None)


def test_prepared_retriever_rejects_more_results_than_top_k() -> None:
    facts = (_fact("fact-1"), _fact("fact-2"))

    class FakeMemory:
        embedding_model = None
        vector_store = SimpleNamespace(
            last_search_metrics=SearchMetrics(search_time_ms=1.0, vector_backend="jasper")
        )

        def search(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "results": [
                    {"memory": fact.text, "score": 0.9, "metadata": {"fact_id": fact.id}}
                    for fact in facts
                ]
            }

    retriever = PreparedMem0Retriever(
        _built(FakeMemory(), facts), sample_id="sample-1", vector_backend="jasper"
    )

    with pytest.raises(RuntimeError, match="returned 2 facts for top_k=1"):
        retriever.search("question", top_k=1)


def test_prepared_retriever_reports_store_search_time_and_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fact = _fact("fact-1")
    memory = SimpleNamespace(
        embedding_model=None,
        vector_store=SimpleNamespace(
            last_search_metrics=SearchMetrics(search_time_ms=7.0, vector_backend="qdrant")
        ),
        search=lambda *_args, **_kwargs: {
            "results": [{"memory": fact.text, "score": 0.9, "metadata": {"fact_id": fact.id}}],
        },
    )
    clock = iter([1.0, 1.009])
    monkeypatch.setattr(
        "benchmarks.memory.mem0.prepared_retriever.time.perf_counter", lambda: next(clock)
    )
    retriever = PreparedMem0Retriever(
        _built(memory, (fact,)), sample_id="sample-1", vector_backend="qdrant"
    )

    hits, metrics = retriever.search("question", top_k=1)

    assert [hit.id for hit in hits] == [fact.id]
    assert metrics.vector_backend == "qdrant"
    assert metrics.search_time_ms == 7.0
    assert metrics.total_time_ms == pytest.approx(9.0)
    assert metrics.embedding_time_ms == 0.0
