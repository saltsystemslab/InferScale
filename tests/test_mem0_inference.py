from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locomo_jasper_bench.config import BenchmarkConfig
from locomo_jasper_bench.data import ConversationSample, Turn
from locomo_jasper_bench.retrieval.fact_catalog import (
    FactCatalogStore,
    MemoryFact,
    fact_catalog_hits,
    locomo_timestamp,
    locomo_turn_role,
)
from locomo_jasper_bench.retrieval.mem0_provider import build_mem0_config, create_mem0_memory
from locomo_jasper_bench.retrieval.memory_builder import (
    SampleMemoryBuilder,
    _mem0_observation_date,
)
from locomo_jasper_bench.retrieval.prepared_retriever import PreparedMem0Retriever
from locomo_jasper_bench.vector_types import VECTOR_DISTANCE, VectorStoreConfig


class _RecordingLlm:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def generate_response(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        self.calls.append((messages, args, kwargs))
        return '{"memory":[{"text":"Alice likes tea."}]}'


class _FakeMemory:
    def __init__(self) -> None:
        self.llm = _RecordingLlm()
        self.add_calls: list[tuple[Any, dict[str, Any]]] = []
        self.vector_store = SimpleNamespace(
            config=SimpleNamespace(backend="qdrant"),
            memory_stats=lambda: {},
        )

    def add(self, messages: Any, **kwargs: Any) -> dict[str, list[dict[str, str]]]:
        self.add_calls.append((messages, kwargs))
        if kwargs.get("infer") is True:
            self.llm.generate_response(messages, response_format={"type": "json_object"})
            return {"results": [{"id": "inferred", "memory": "Alice likes tea."}]}
        return {
            "results": [
                {
                    "id": kwargs.get("metadata", {}).get("fact_id", "raw"),
                    "memory": messages[0]["content"],
                }
            ]
        }


class _DeterministicEmbedder:
    def embed(self, text: Any, purpose: str | None = None) -> list[float]:
        del text, purpose
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts: list[Any], purpose: str | None = None) -> list[list[float]]:
        return [self.embed(text, purpose) for text in texts]


def _sample() -> ConversationSample:
    return ConversationSample(
        sample_id="sample-1",
        turns=[
            Turn(
                sample_id="sample-1",
                session_id="session_1",
                session_index=1,
                turn_index=0,
                speaker="Alice",
                text="I like tea.",
                timestamp="2026-01-02",
            )
        ],
        qa=[],
        raw={"conversation": {"speaker_a": "Alice", "speaker_b": "Bob"}},
    )


def test_mem0_config_uses_explicit_memory_llm_settings(tmp_path: Path) -> None:
    config = build_mem0_config(
        store_root=tmp_path,
        vector_config=VectorStoreConfig(backend="qdrant"),
        embedding_model="embedding-model",
        embedding_api_key="embedding-secret",
        embedding_base_url="https://embeddings.example/v1",
        memory_llm_provider="vllm",
        memory_llm_model="Qwen/Qwen2.5-7B-Instruct",
        memory_llm_api_key="memory-secret",
        memory_llm_base_url="https://memory.example/v1",
    )

    assert config["llm"] == {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "temperature": 0.0,
            "api_key": "memory-secret",
            "vllm_base_url": "https://memory.example/v1",
        },
    }


def test_locomo_roles_and_timestamps_match_pinned_memory_benchmark() -> None:
    sample = _sample()
    speaker_a_turn = sample.turns[0]
    speaker_b_turn = Turn(
        sample_id="sample-1",
        session_id="session_1",
        session_index=1,
        turn_index=1,
        speaker="Bob",
        text="I prefer coffee.",
        timestamp="1:56 pm on 8 May, 2023",
    )

    assert locomo_turn_role(sample, speaker_a_turn) == "user"
    assert locomo_turn_role(sample, speaker_b_turn) == "assistant"
    assert locomo_timestamp("1:56 pm on 8 May, 2023") == (
        1683554160,
        "2023-05-08T13:56:00+00:00",
    )
    assert locomo_timestamp("1:56 pm on 8 May, 2023")[0] == locomo_timestamp(
        "1:56 pm on 8 May, 2023"
    )[0]
    assert locomo_timestamp("1:56 pm on 8 Sep, 2023") == (
        1694181360,
        "2023-09-08T13:56:00+00:00",
    )

    with pytest.raises(ValueError, match="Unsupported LoCoMo timestamp"):
        locomo_timestamp("sometime last spring")


def test_mem0_observation_date_shim_anchors_prompt_and_restores_global(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0-global"))
    monkeypatch.setenv("MEM0_TELEMETRY", "false")
    import mem0

    del mem0
    from mem0.memory import main as mem0_main

    original = mem0_main.generate_additive_extraction_prompt
    with _mem0_observation_date("2026-01-02T00:00:00+00:00"):
        prompt = mem0_main.generate_additive_extraction_prompt(
            existing_memories=[],
            new_messages="user: Alice likes tea.",
            last_k_messages=[],
        )

    assert "## Observation Date\n2026-01-02" in prompt
    assert mem0_main.generate_additive_extraction_prompt is original


def test_fact_catalog_hits_promote_stable_source_metadata() -> None:
    fact = MemoryFact(
        id="fact-1",
        text="Alice likes tea.",
        created_at="2026-01-02T00:00:00+00:00",
        timestamp_epoch=1767312000,
        sample_id="sample-1",
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=0,
        source_turn_id="sample-1:session_1:0",
        speaker="Alice",
        role="user",
    )

    hit = fact_catalog_hits([fact])[0]

    assert hit.id == "fact-1"
    assert hit.payload["created_at"] == fact.created_at
    assert hit.payload["source_session_index"] == 1
    assert hit.payload["source_session_id"] == "session_1"
    assert hit.payload["source_turn_id"] == "sample-1:session_1:0"
    assert hit.payload["metadata"]["fact_id"] == "fact-1"


def test_fact_catalog_identity_includes_mem0_embedding_and_endpoints(tmp_path: Path) -> None:
    sample = _sample()
    fact = MemoryFact(
        id="fact-1",
        text="Alice likes tea.",
        created_at="2026-01-02T00:00:00+00:00",
        timestamp_epoch=1767312000,
        sample_id="sample-1",
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=0,
        source_turn_id="sample-1:session_1:0",
        speaker="Alice",
        role="user",
    )
    store = FactCatalogStore(
        tmp_path,
        provider="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        endpoint="https://memory.example/v1",
        embedding_model="embedding-model",
        embedding_endpoint="https://embedding.example/v1",
        mem0_version="2.0.11",
    )
    store.write(sample, [fact])

    assert store.load(sample) == (fact,)
    different_endpoint = FactCatalogStore(
        tmp_path,
        provider="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        endpoint="https://other.example/v1",
        embedding_model="embedding-model",
        embedding_endpoint="https://embedding.example/v1",
        mem0_version="2.0.11",
    )
    assert different_endpoint.path_for(sample) != store.path_for(sample)
    with pytest.raises(RuntimeError, match="Missing Mem0 fact catalog"):
        different_endpoint.load(sample)


def test_fact_catalog_identity_requires_current_inner_product_and_temperature(tmp_path: Path) -> None:
    sample = _sample()
    fact = MemoryFact(
        id="fact-1",
        text="Alice likes tea.",
        created_at="2026-01-02T00:00:00+00:00",
        timestamp_epoch=1767312000,
        sample_id="sample-1",
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=0,
        source_turn_id="sample-1:session_1:0",
        speaker="Alice",
        role="user",
    )
    store_kwargs = dict(
        provider="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        endpoint="https://memory.example/v1",
        embedding_model="embedding-model",
        embedding_endpoint="https://embedding.example/v1",
        mem0_version="2.0.11",
    )
    baseline = FactCatalogStore(tmp_path, **store_kwargs)
    baseline.write(sample, [fact])

    payload = json.loads(baseline.path_for(sample).read_text(encoding="utf-8"))
    assert payload["version"] == 4
    assert payload["vector_distance"] == VECTOR_DISTANCE

    payload.pop("memory_llm_temperature")
    baseline.path_for(sample).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="memory_llm_temperature"):
        baseline.load(sample)

    baseline.write(sample, [fact])
    warmer = FactCatalogStore(tmp_path, temperature=0.1, **store_kwargs)
    with pytest.raises(RuntimeError, match="memory_llm_temperature"):
        warmer.load(sample)

    payload = json.loads(baseline.path_for(sample).read_text(encoding="utf-8"))
    payload.pop("vector_distance")
    baseline.path_for(sample).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="vector_distance"):
        baseline.load(sample)

    baseline.write(sample, [fact])
    payload = json.loads(baseline.path_for(sample).read_text(encoding="utf-8"))
    payload["vector_distance"] = "cosine"
    baseline.path_for(sample).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="vector_distance"):
        baseline.load(sample)


def test_prepared_retriever_uses_live_mem0_search_and_returns_best_first_facts() -> None:
    fact = MemoryFact(
        id="fact-1",
        text="Alice likes tea.",
        created_at="2026-01-02T00:00:00+00:00",
        timestamp_epoch=1767312000,
        sample_id="sample-1",
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=0,
        source_turn_id="sample-1:session_1:0",
        speaker="Alice",
        role="user",
    )

    class SearchMemory:
        def __init__(self) -> None:
            self.embedding_model = _DeterministicEmbedder()
            self.vector_store = SimpleNamespace(
                last_search_metrics=SimpleNamespace(
                    search_time_ms=2.5,
                    vector_backend="qdrant",
                    jasper_effective_beam_width=None,
                )
            )
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((query, kwargs))
            self.embedding_model.embed(query, "search")
            return {
                "results": [
                    {
                        "id": "raw-mem0-id",
                        "memory": fact.text,
                        "created_at": fact.created_at,
                        "score": 0.91,
                        "metadata": fact.metadata(),
                    }
                ]
            }

    memory = SearchMemory()
    retriever = PreparedMem0Retriever(
        memory,
        sample_id="sample-1",
        fact_catalog=(fact,),
        vector_backend="qdrant",
    )

    hits, metrics = retriever.search("tea", top_k=5)

    assert memory.calls == [
        ("tea", {"top_k": 5, "filters": {"user_id": "sample-1"}})
    ]
    assert [hit.id for hit in hits] == ["fact-1"]
    assert hits[0].rank == 1
    assert hits[0].payload["source_session_index"] == 1
    assert metrics.vector_backend == "qdrant"
    assert metrics.search_time_ms == 2.5


def test_sample_builder_materializes_catalog_and_read_mode_never_reruns_inference(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    created: list[tuple[_FakeMemory, dict[str, Any]]] = []

    def fake_create_mem0_memory(**kwargs: Any) -> _FakeMemory:
        memory = _FakeMemory()
        created.append((memory, kwargs))
        return memory

    monkeypatch.setattr(
        "locomo_jasper_bench.retrieval.memory_builder.create_mem0_memory",
        fake_create_mem0_memory,
    )
    config = BenchmarkConfig(
        results_dir=tmp_path / "results",
        run_id="run",
        model="Qwen/Qwen2.5-7B-Instruct",
        vector_backend="qdrant",
        embedding_cache_enabled=False,
        memory_llm_api_key="memory-secret",
        memory_llm_cache_dir=tmp_path / "inference-cache",
    )

    writer = SampleMemoryBuilder(config, memory_llm_cache_mode="write")
    _, write_metrics = writer.build_with_metrics(_sample(), finalize_index=False)

    write_memory, create_kwargs = created[0]
    assert create_kwargs["memory_llm_provider"] == "vllm"
    assert create_kwargs["memory_llm_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert create_kwargs["memory_llm_api_key"] == "memory-secret"
    assert len(write_memory.llm._wrapped.calls) == 1
    assert write_metrics["memory_input_turn_count"] == 1
    assert write_metrics["memory_inferred_record_count"] == 1
    assert write_metrics["memory_llm_cache_hits"] == 0
    assert write_metrics["memory_llm_cache_misses"] == 1

    messages, add_kwargs = write_memory.add_calls[0]
    assert messages == [{"role": "user", "content": "Alice: I like tea."}]
    assert add_kwargs["infer"] is True
    assert add_kwargs["user_id"] == "sample-1"
    assert add_kwargs["metadata"]["created_at"] == "2026-01-02T00:00:00+00:00"
    assert add_kwargs["metadata"]["timestamp"] == 1767312000
    assert add_kwargs["metadata"]["timestamp_epoch"] == 1767312000
    assert add_kwargs["metadata"]["source_session_index"] == 1
    assert add_kwargs["metadata"]["source_session_id"] == "session_1"
    assert add_kwargs["metadata"]["source_turn_id"] == "sample-1:session_1:0"
    assert add_kwargs["metadata"]["role"] == "user"
    assert "timestamp" not in add_kwargs

    facts = writer.load_fact_catalog(_sample())
    assert len(facts) == 1
    assert facts[0].text == "Alice likes tea."
    assert facts[0].created_at == "2026-01-02T00:00:00+00:00"
    assert facts[0].source_turn_id == "sample-1:session_1:0"

    reader = SampleMemoryBuilder(config)
    _, read_metrics = reader.build_with_metrics(_sample(), finalize_index=False)

    read_memory, _ = created[1]
    assert read_memory.llm.calls == []
    assert read_memory.add_calls[0][1]["infer"] is False
    assert read_metrics["memory_fact_catalog_loaded"] == 1
    assert read_metrics["memory_inferred_record_count"] == 1


def test_mem0_2_inference_persists_turn_metadata_through_custom_adapter(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0-global"))
    monkeypatch.setenv("MEM0_TELEMETRY", "false")
    memory = create_mem0_memory(
        store_root=tmp_path / "store",
        vector_config=VectorStoreConfig(backend="qdrant"),
        embedding_model="text-embedding-3-small",
        embedding_api_key="test-key",
        embedding_base_url=None,
        memory_llm_provider="vllm",
        memory_llm_model="Qwen/Qwen2.5-7B-Instruct",
        memory_llm_api_key="test-key",
        memory_llm_base_url=None,
    )
    memory.llm = _RecordingLlm()
    memory.embedding_model = _DeterministicEmbedder()

    try:
        result = memory.add(
            [{"role": "user", "content": "Alice: I like tea."}],
            user_id="sample-1",
            infer=True,
            metadata={
                "sample_id": "sample-1",
                "turn_id": "sample-1:session_1:0",
                "fact_id": "fact-1",
            },
        )
        search_result = memory.search(
            "tea",
            top_k=1,
            filters={"user_id": "sample-1"},
        )
    finally:
        memory.vector_store.close()

    assert result["results"][0]["memory"] == "Alice likes tea."
    rows = search_result["results"]
    assert len(rows) == 1
    assert rows[0]["id"] == "fact-1"
    assert rows[0]["memory"] == "Alice likes tea."
    assert rows[0]["metadata"]["turn_id"] == "sample-1:session_1:0"
