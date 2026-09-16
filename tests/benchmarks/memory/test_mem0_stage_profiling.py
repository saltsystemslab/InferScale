from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.memory.mem0.adapter import Mem0JasperVectorStore
from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.mem0.prepared_retriever import PreparedMem0Retriever
from benchmarks.memory.mem0.profiling import RETRIEVAL_STAGE_METRIC_KEYS

mem0_main = pytest.importorskip("mem0.memory.main")


class _Memory:
    """Use the installed Mem0 retrieval algorithm with local, deterministic inputs."""

    _normalize_entity_text = staticmethod(mem0_main.Memory._normalize_entity_text)

    def __init__(self, primary, entities):
        self.vector_store = primary
        self._entity_store = self.entity_store = entities
        self.embedding_model = SimpleNamespace(
            embed=lambda text, action: [1.0, 0.0],
            embed_batch=lambda texts, action: [[1.0, 0.0] for _ in texts],
        )

    def _compute_entity_boosts(self, *args, **kwargs):
        return mem0_main.Memory._compute_entity_boosts(self, *args, **kwargs)

    def search(self, query, *, top_k, filters):
        return {"results": mem0_main.Memory._search_vector_store(self, query, filters, top_k)}


def _retriever(tmp_path, backend):
    primary = Mem0JasperVectorStore(path=tmp_path, collection_name="facts", backend=backend)
    entities = Mem0JasperVectorStore(path=tmp_path, collection_name="entities", backend=backend)
    facts = tuple(MemoryFact(
        id=f"fact-{i}", text=text, created_at="2023-01-01T00:00:00+00:00",
        timestamp_epoch=1672531200, sample_id="sample", source_session_index=1,
        source_session_id="session_1", source_turn_index=i,
        source_turn_id=f"sample:session_1:{i}", speaker="Alice", role="user",
    ) for i, text in enumerate(["Alice lives in Paris.", "Bob lives in Rome."]))
    primary.insert(
        [[1.0, 0.0], [0.7, 0.3]],
        [{"data": fact.text, **fact.metadata()} for fact in facts],
        [fact.id for fact in facts],
    )
    entities.insert(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        [{"data": text, "user_id": "sample", "linked_memory_ids": [facts[0].id]}
         for text in ["Alice", "Paris", "Elsewhere"]],
        ["alice", "paris", "other"],
    )
    memory = _Memory(primary, entities)
    return PreparedMem0Retriever(memory, sample_id="sample", fact_catalog=facts, vector_backend=backend)


@pytest.mark.parametrize("backend", ["qdrant", "jasper"])
@pytest.mark.parametrize("stages,copies", [(False, False), (False, True), (True, False), (True, True)])
def test_real_mem0_pipeline_stage_diagnostics_preserve_results_and_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, stages: bool, copies: bool,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1" if stages else "0")
    monkeypatch.setenv("QDRANT_PROFILE_DEEPCOPY", "1" if copies else "0")
    monkeypatch.setattr(mem0_main, "lemmatize_for_bm25", lambda text: text.lower())
    monkeypatch.setattr(mem0_main, "extract_entities", lambda text: [] if text == "none" else [
        ("PERSON", "Alice"), ("GPE", "Paris"),
    ])
    retriever = _retriever(tmp_path, backend)
    original_embedder = retriever.memory.embedding_model
    original_extract = mem0_main.extract_entities
    original_pipeline = mem0_main.Memory._search_vector_store
    try:
        first_hits = None
        for query in ["Alice in Paris?", "none", "Alice in Paris?"]:
            hits, metrics = retriever.search(query, top_k=1)
            assert hits[0].id == "fact-0"
            if query != "none":
                if first_hits is None:
                    first_hits = hits
                else:
                    assert hits == first_hits
            assert retriever.memory.embedding_model is original_embedder
            assert retriever.memory.vector_store._retrieval_profile is None
            assert retriever.memory._entity_store._retrieval_profile is None
            assert mem0_main.extract_entities is original_extract
            assert mem0_main.Memory._search_vector_store is original_pipeline
            data = metrics.stage_timings
            if not stages:
                assert data is None
                continue
            assert set(data) <= set(RETRIEVAL_STAGE_METRIC_KEYS)
            assert all(value >= 0 for value in data.values())
            assert data["mem0_primary_search_calls"] == 1
            assert data["mem0_entity_search_calls"] == (0 if query == "none" else 2)
            assert data["mem0_primary_search_time_ms"] >= data["mem0_primary_backend_search_time_ms"]
            assert data["mem0_entity_search_time_ms"] >= data["mem0_entity_search_wall_time_ms"]
            assert data["mem0_entity_boost_time_ms"] >= data["mem0_entity_search_wall_time_ms"]
            assert data["mem0_entity_search_wall_time_ms"] >= data["mem0_entity_backend_search_wall_time_ms"]
            if query == "none":
                for field in ("mem0_entity_embedding_time_ms", "mem0_entity_boost_time_ms",
                              "mem0_entity_search_time_ms", "mem0_entity_search_wall_time_ms"):
                    assert data[field] == 0
            top_level = ("lemmatization", "entity_extraction", "query_embedding", "primary_search",
                         "keyword_search", "entity_boost", "candidate_build", "ranking", "result_format")
            covered = sum(data[f"mem0_{stage}_time_ms"] for stage in top_level)
            assert covered + data["mem0_request_residual_time_ms"] <= metrics.total_time_ms
            assert data["mem0_result_format_time_ms"] > 0
            assert data["mem0_result_conversion_time_ms"] > 0
            if backend == "qdrant":
                assert "mem0_primary_jasper_graph_search_time_ms" not in data
                assert data["mem0_primary_qdrant_scoring_time_ms"] > 0
                assert data["mem0_primary_qdrant_filter_time_ms"] > 0
                assert data["mem0_primary_qdrant_point_construction_time_ms"] > 0
                if copies:
                    assert metrics.qdrant_deepcopy_calls == 2
                    assert data["mem0_primary_payload_access_time_ms"] >= metrics.qdrant_deepcopy_time_ms
                    assert data["mem0_entity_payload_access_wall_time_ms"] >= metrics.qdrant_entity_deepcopy_wall_time_ms
            else:
                assert "mem0_primary_qdrant_scoring_time_ms" not in data
                assert data["mem0_primary_jasper_exact_cpu_search_time_ms"] > 0
                assert data["mem0_primary_jasper_graph_search_time_ms"] == 0
        # Compare the same live stores with diagnostics disabled, not reconstructed data.
        monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "0")
        baseline_hits, baseline_metrics = retriever.search("Alice in Paris?", top_k=1)
        assert baseline_hits == first_hits
        assert baseline_metrics.stage_timings is None
    finally:
        retriever.close()


@pytest.mark.parametrize("backend", ["qdrant", "jasper"])
def test_entity_failure_caught_by_mem0_is_profiled_and_later_queries_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    monkeypatch.setenv("MEM0_PROFILE_RETRIEVAL", "1")
    monkeypatch.setattr(mem0_main, "lemmatize_for_bm25", lambda text: text)
    monkeypatch.setattr(mem0_main, "extract_entities", lambda text: [("PERSON", "Alice")])
    retriever = _retriever(tmp_path, backend)
    store = retriever.memory._entity_store.store
    original = store.search

    def fail(*args, **kwargs):
        raise ValueError("entity backend failure")

    try:
        monkeypatch.setattr(store, "search", fail)
        hits, metrics = retriever.search("Alice", top_k=1)
        assert hits[0].id == "fact-0"
        assert metrics.stage_timings["mem0_entity_search_calls"] == 1
        assert metrics.stage_timings["mem0_entity_backend_search_time_ms"] > 0
        assert metrics.stage_timings["mem0_entity_adapter_count_time_ms"] == 0
        monkeypatch.setattr(store, "search", original)
        _, next_metrics = retriever.search("Alice", top_k=1)
        assert next_metrics.stage_timings["mem0_entity_adapter_count_time_ms"] > 0
    finally:
        retriever.close()
