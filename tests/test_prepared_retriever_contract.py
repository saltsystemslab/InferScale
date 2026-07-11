from __future__ import annotations

from types import SimpleNamespace

import pytest

from locomo_jasper_bench.retrieval.fact_catalog import MemoryFact
from locomo_jasper_bench.retrieval.prepared_retriever import PreparedMem0Retriever
from locomo_jasper_bench.vector_types import SearchMetrics


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
