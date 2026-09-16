from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.mem0.prepared_retriever import PreparedMem0Retriever
from benchmarks.common.vector_types import SearchMetrics


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


def test_prepared_retriever_rejects_more_results_than_top_k() -> None:
    facts = (_fact("fact-1"), _fact("fact-2"))

    class FakeMemory:
        embedding_model = None
        vector_store = SimpleNamespace(
            last_search_metrics=SearchMetrics(
                search_time_ms=1.0,
                vector_backend="jasper",
            )
        )

        def search(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "results": [
                    {
                        "memory": fact.text,
                        "score": 0.9,
                        "metadata": {"fact_id": fact.id},
                    }
                    for fact in facts
                ]
            }

    retriever = PreparedMem0Retriever(
        FakeMemory(),
        sample_id="sample-1",
        fact_catalog=facts,
        vector_backend="jasper",
    )

    with pytest.raises(RuntimeError, match="returned 2 facts for top_k=1"):
        retriever.search("question", top_k=1)


@pytest.mark.parametrize("metrics_type", [SearchMetrics, SimpleNamespace])
@pytest.mark.parametrize("copy_ms,copy_calls,query_copy_ms,query_copy_calls", [
    (None, None, None, None),
    (0.0, 0, 0.0, 0),
    (1.25, 2, 0.375, 1),
    (None, None, 0.375, 1),
    (1.25, 2, None, None),
])
def test_prepared_retriever_preserves_optional_deepcopy_diagnostics_and_existing_timings(
    monkeypatch: pytest.MonkeyPatch, metrics_type: type,
    copy_ms: float | None, copy_calls: int | None,
    query_copy_ms: float | None, query_copy_calls: int | None,
) -> None:
    fact = _fact("fact-1")
    store_metrics = metrics_type(
        search_time_ms=7.0,
        vector_backend="qdrant",
        qdrant_deepcopy_time_ms=copy_ms,
        qdrant_deepcopy_calls=copy_calls,
        qdrant_query_deepcopy_time_ms=query_copy_ms,
        qdrant_query_deepcopy_calls=query_copy_calls,
    )
    memory = SimpleNamespace(
        embedding_model=None,
        vector_store=SimpleNamespace(last_search_metrics=store_metrics),
        search=lambda *_args, **_kwargs: {
            "results": [{"memory": fact.text, "score": 0.9, "metadata": {"fact_id": fact.id}}],
        },
    )
    clock = iter([1.0, 1.009])
    monkeypatch.setattr(
        "benchmarks.memory.mem0.prepared_retriever.time.perf_counter", lambda: next(clock),
    )
    retriever = PreparedMem0Retriever(
        memory, sample_id="sample-1", fact_catalog=(fact,), vector_backend="qdrant",
    )

    hits, metrics = retriever.search("question", top_k=1)

    assert [hit.id for hit in hits] == [fact.id]
    assert metrics.qdrant_deepcopy_time_ms == copy_ms
    assert metrics.qdrant_deepcopy_calls == copy_calls
    assert metrics.qdrant_query_deepcopy_time_ms == query_copy_ms
    assert metrics.qdrant_query_deepcopy_calls == query_copy_calls
    assert metrics.qdrant_entity_deepcopy_time_ms is None
    assert metrics.qdrant_entity_deepcopy_wall_time_ms is None
    assert metrics.qdrant_entity_deepcopy_calls is None
    assert metrics.qdrant_entity_query_deepcopy_time_ms is None
    assert metrics.qdrant_entity_query_deepcopy_wall_time_ms is None
    assert metrics.qdrant_entity_query_deepcopy_calls is None
    assert metrics.search_time_ms == 7.0
    assert metrics.total_time_ms == pytest.approx(9.0)
    assert metrics.embedding_time_ms == 0.0


def test_prepared_retriever_accepts_legacy_metrics_without_query_copy_fields() -> None:
    fact = _fact("fact-1")
    memory = SimpleNamespace(
        embedding_model=None,
        vector_store=SimpleNamespace(last_search_metrics=SimpleNamespace(
            search_time_ms=7.0, vector_backend="qdrant",
            qdrant_deepcopy_time_ms=1.25, qdrant_deepcopy_calls=2,
        )),
        search=lambda *_args, **_kwargs: {
            "results": [{"memory": fact.text, "score": 0.9, "metadata": {"fact_id": fact.id}}],
        },
    )
    retriever = PreparedMem0Retriever(
        memory, sample_id="sample-1", fact_catalog=(fact,), vector_backend="qdrant",
    )

    _, metrics = retriever.search("question", top_k=1)

    assert metrics.qdrant_deepcopy_time_ms == 1.25
    assert metrics.qdrant_deepcopy_calls == 2
    assert metrics.qdrant_query_deepcopy_time_ms is None
    assert metrics.qdrant_query_deepcopy_calls is None
