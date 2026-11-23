from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.common.vector_types import RetrievalMetrics, SearchMetrics
from benchmarks.memory.data import ConversationSample, Turn
from benchmarks.memory.mem0 import prepared_retriever
from benchmarks.memory.mem0.fact_catalog import make_memory_fact
from benchmarks.memory.mem0.memory_handles import BuiltMemory
from benchmarks.memory.mem0.prepared_retriever import PreparedMem0Retriever

QDRANT_METRICS = SearchMetrics(search_time_ms=2.5, vector_backend="qdrant")


def catalog_facts():
    turns = [
        Turn("sample-1", "session_1", 1, index, "Alice", f"Turn {index}", "2026-01-02")
        for index in range(3)
    ]
    sample = ConversationSample("sample-1", turns, [], {})
    return tuple(make_memory_fact(f"Fact {turn.turn_index}.", sample, turn) for turn in turns)


def mem0_row(fact, **changes):
    row = {"id": "mem0-id", "memory": fact.text, "score": 0.75, "metadata": fact.metadata()}
    return {**row, **changes}


class SearchableMemory:
    def __init__(self, result, store_metrics=QDRANT_METRICS, embedding_ms_per_search=0.0):
        self.result = result
        self.searches = []
        self.embedder = SimpleNamespace(embed_time_ms=10.0)
        self.embedding_ms_per_search = embedding_ms_per_search
        self.vector_store = SimpleNamespace(last_search_metrics=store_metrics, close=self.closed)
        self._entity_store = None
        self.events = []

    def search(self, query, **options):
        self.searches.append((query, options))
        self.embedder.embed_time_ms += self.embedding_ms_per_search
        return self.result

    def closed(self):
        self.events.append("vector store closed")


def retriever_over(memory, facts, *, vector_backend="qdrant", embedder=None):
    built = BuiltMemory(memory=memory, facts=facts, metrics={}, embedder=embedder)
    return PreparedMem0Retriever(built, sample_id="sample-1", vector_backend=vector_backend)


def test_search_asks_mem0_for_the_sample_and_returns_catalog_facts_best_first(monkeypatch):
    first, second, _ = facts = catalog_facts()
    memory = SearchableMemory(
        {"results": [mem0_row(second), mem0_row(first, score=0.5)]}, embedding_ms_per_search=4.0
    )
    clock = iter([100.0, 100.012])
    monkeypatch.setattr(prepared_retriever.time, "perf_counter", lambda: next(clock))
    retriever = retriever_over(memory, facts, embedder=memory.embedder)

    hits, metrics = retriever.search("Where?", top_k=2)

    assert memory.searches == [("Where?", {"top_k": 2, "filters": {"user_id": "sample-1"}})]
    assert [(hit.id, hit.rank, hit.score, hit.distance) for hit in hits] == [
        (second.id, 1, 0.75, 0.25),
        (first.id, 2, 0.5, 0.5),
    ]
    assert isinstance(metrics, RetrievalMetrics)
    assert metrics.embedding_time_ms == 4.0
    assert metrics.search_time_ms == 2.5
    assert metrics.total_time_ms == pytest.approx(12.0)
    assert (metrics.vector_backend, metrics.jasper_effective_beam_width) == ("qdrant", None)
