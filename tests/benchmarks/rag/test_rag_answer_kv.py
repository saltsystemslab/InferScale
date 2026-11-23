from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from library_fakes import FakeTokenizer

from benchmarks.rag import answer_kv
from benchmarks.rag.answer_kv import RagKvAnswerClient
from benchmarks.rag.corpus import load_rag_corpus
from benchmarks.rag.kv_cache import CachedRagScaffold, RagChunkCache
from benchmarks.rag.prompting import extract_rag_scaffold_token_ids
from inferscale.v1.serving.vllm import build_kv_transfer_config
from inferscale.v1.types import KVChunk, SearchHit


class ChunkStoreDouble:
    def __init__(self, events: list[Any], chunks) -> None:
        self._events = events
        self._encoded = {
            chunk.chunk_id: KVChunk(chunk.chunk_id, list(chunk.token_ids), {}) for chunk in chunks
        }

    def fetch(self, chunk_ids: list[str]) -> list[KVChunk]:
        self._events.append(("fetch", list(chunk_ids)))
        return [self._encoded[chunk_id] for chunk_id in chunk_ids]


@pytest.fixture
def kv_serving(monkeypatch, multihop_config, engine):
    config = multihop_config(run_id="kv-run")
    corpus = load_rag_corpus(config)
    events: list[Any] = []
    scaffold = extract_rag_scaffold_token_ids(
        FakeTokenizer(), system_prompt=corpus.spec.prompt_profile.system_prompt, block_size=16
    )
    cached_scaffold = CachedRagScaffold(
        cos_table="cached cos table",
        sin_table="cached sin table",
        scaffold_chunks={
            "header": KVChunk("header", scaffold.header_token_ids, {}),
            "empty_passages": KVChunk("empty_passages", scaffold.empty_token_ids, {}),
            "footer": KVChunk("footer", scaffold.footer_token_ids, {}),
        },
    )

    class Clock:
        def perf_counter(self) -> float:
            events.append("clock")
            return float(events.count("clock"))

    class ComposeOnlyEncoder:
        def __init__(self, **_: Any) -> None:
            raise AssertionError("answer runs must not load the encoder model weights")

        @classmethod
        def from_tables(cls, **tables: Any) -> ComposeOnlyEncoder:
            events.append(("encoder from tables", tables))
            return object.__new__(cls)

        def compose(self, parts: list[KVChunk]) -> dict[str, Any]:
            events.append(("compose", [part.chunk_id for part in parts]))
            return {"layer": "composed kv"}

    def build_store(cache, *, chunks, device) -> ChunkStoreDouble:
        events.append(("store loaded", device))
        return ChunkStoreDouble(events, chunks)

    def start_engine(*, kv_transfer_config=None) -> float:
        events.append("engine started")
        engine.started = True
        engine.started_with = dict(kv_transfer_config)
        return 0.0

    generate = engine.generate

    def generate_and_record(prompt_token_ids, **sampling):
        events.append("generate")
        return generate(prompt_token_ids, **sampling)

    def register_user_memory(namespace: str, **memory: Any) -> None:
        events.append(("register", namespace, memory))

    def namespace_stats(namespace: str) -> dict[str, Any]:
        events.append(("stats", namespace))
        return {"total_gpu_mb": 12.5}

    monkeypatch.setattr(engine, "start", start_engine)
    monkeypatch.setattr(engine, "generate", generate_and_record)
    monkeypatch.setattr(answer_kv, "time", Clock())
    monkeypatch.setattr(answer_kv, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="1d")))
    monkeypatch.setattr(answer_kv, "load_encoder_tokenizer", lambda model: FakeTokenizer())
    monkeypatch.setattr(answer_kv, "load_tables_and_scaffold", lambda *_, **__: cached_scaffold)
    monkeypatch.setattr(answer_kv, "ChunkedRopeEncoder", ComposeOnlyEncoder)
    monkeypatch.setattr(answer_kv, "CpuChunkStore", build_store)
    monkeypatch.setattr(
        answer_kv,
        "get_gpu_memory_store",
        lambda namespace, **store: events.append(("namespace created", namespace, store)),
    )
    monkeypatch.setattr(answer_kv, "register_user_memory", register_user_memory)
    monkeypatch.setattr(answer_kv, "namespace_stats", namespace_stats)
    monkeypatch.setattr(
        answer_kv,
        "remove_user_memory",
        lambda namespace, user_id: events.append(("remove", namespace, user_id)),
    )
    return SimpleNamespace(
        config=config, corpus=corpus, events=events, scaffold=scaffold, engine=engine
    )


def build_client(kv_serving) -> RagKvAnswerClient:
    return RagKvAnswerClient(
        kv_serving.config,
        chunks=kv_serving.corpus.chunks,
        cache=RagChunkCache.for_config(kv_serving.config, kv_serving.corpus.docs),
        prompt_profile=kv_serving.corpus.spec.prompt_profile,
    )


def started_client(kv_serving) -> RagKvAnswerClient:
    client = build_client(kv_serving)
    client.start_llm()
    kv_serving.events.clear()
    return client


def hits_for(*chunk_ids: str) -> list[SearchHit]:
    return [
        SearchHit(id=chunk_id, payload={}, score=1.0, distance=0.0, rank=rank)
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def names(events: list[Any]) -> list[str]:
    return [event if isinstance(event, str) else event[0] for event in events]


def test_answer_runs_compose_from_cached_rope_tables_without_loading_encoder_weights(kv_serving):
    build_client(kv_serving)

    from_tables = kv_serving.events[0]
    assert from_tables[0] == "encoder from tables"
    assert {name: value for name, value in from_tables[1].items() if name != "tokenizer"} == {
        "model": "llama",
        "device": "cpu",
        "max_position": 32768,
        "cos_table": "cached cos table",
        "sin_table": "cached sin table",
    }
    assert isinstance(from_tables[1]["tokenizer"], FakeTokenizer)


def test_composed_memory_is_handed_to_the_connector_through_a_gpu_resident_store(kv_serving):
    client = build_client(kv_serving)
    client.start_llm()

    assert client.namespace == "kv-run-1d"
    assert client.active_user_id == "kv-run-1d-active"
    assert kv_serving.events[1] == ("namespace created", "kv-run-1d", {"backend": "gpu"})
    assert kv_serving.engine.started_with == build_kv_transfer_config(
        connector_module="inferscale.v1.kv.connector",
        namespace="kv-run-1d",
        default_user_id="kv-run-1d-active",
        store_backend="gpu",
    )
    assert (
        kv_serving.engine.started_with["kv_connector_extra_config"]["memory_store_backend"] == "gpu"
    )


def test_answer_times_each_step_and_generates_while_the_memory_is_registered(kv_serving):
    client = started_client(kv_serving)
    first, second, third = (chunk.chunk_id for chunk in kv_serving.corpus.chunks[:3])

    client.answer(kv_serving.corpus.queries[0], hits_for(first, second, third))

    assert names(kv_serving.events) == [
        "clock",
        "clock",
        "fetch",
        "clock",
        "clock",
        "compose",
        "clock",
        "clock",
        "clock",
        "clock",
        "register",
        "clock",
        "stats",
        "generate",
        "remove",
    ]
    by_name = {event[0]: event for event in kv_serving.events if not isinstance(event, str)}
    assert by_name["fetch"] == ("fetch", [third, second, first])
    assert by_name["compose"] == ("compose", ["header", third, second, first, "footer"])
    assert by_name["stats"] == ("stats", "kv-run-1d")
    assert by_name["remove"] == ("remove", "kv-run-1d", "kv-run-1d-active")


def test_gpu_memory_is_measured_while_the_composed_memory_is_still_registered(kv_serving):
    client = started_client(kv_serving)

    answer = client.answer(
        kv_serving.corpus.queries[0], hits_for(kv_serving.corpus.chunks[0].chunk_id)
    )

    registry_calls = [name for name in names(kv_serving.events) if name != "clock"]
    assert registry_calls[-4:] == ["register", "stats", "generate", "remove"]
    assert answer.metrics["kv_store_gpu_mb"] == 12.5


def test_answer_metrics_exclude_parity_check_and_registry_handoff_from_latency(kv_serving):
    client = started_client(kv_serving)
    chunks = kv_serving.corpus.chunks[:3]
    hits = hits_for(*(chunk.chunk_id for chunk in chunks))

    answer = client.answer(kv_serving.corpus.queries[0], hits, query_started_at=0.5)

    scaffold = kv_serving.scaffold
    memory_tokens = len(scaffold.header_token_ids) + 3 * 48 + len(scaffold.footer_token_ids)
    assert memory_tokens == 323
    assert answer.metrics == {
        "kv_memory_tokens": 323,
        "kv_fetch_time_ms": 1000.0,
        "kv_compose_time_ms": 1000.0,
        "kv_verify_time_ms": 1000.0,
        "kv_store_write_time_ms": 1000.0,
        "kv_query_tokens": len(kv_serving.engine.prompts[0]) - 323,
        "kv_query_bos_stripped": 0,
        "kv_selected_chunk_ids": [chunk.chunk_id for chunk in reversed(chunks)],
        "kv_block_size": 16,
        "kv_context_window": 1,
        "kv_loaded_memory_tokens": 320,
        "kv_recomputed_memory_tail_tokens": 3,
        "retrieved_chunk_count": 3,
        "answer_generate_time_ms": 250.0,
        "answer_total_time_ms": 99250.0 - 2000.0,
        "kv_engine_time_to_first_token_ms": 12.0,
        "answer_time_to_first_token_ms": 99000.0 - 2000.0 + 12.0,
        "query_to_answer_ms": 99750.0 - 2000.0,
        "query_to_first_token_ms": 99500.0 - 2000.0 + 12.0,
        "kv_store_gpu_mb": 12.5,
    }
    assert list(answer.metrics) == [
        "kv_memory_tokens",
        "kv_fetch_time_ms",
        "kv_compose_time_ms",
        "kv_verify_time_ms",
        "kv_store_write_time_ms",
        "kv_query_tokens",
        "kv_query_bos_stripped",
        "kv_selected_chunk_ids",
        "kv_block_size",
        "kv_context_window",
        "kv_loaded_memory_tokens",
        "kv_recomputed_memory_tail_tokens",
        "retrieved_chunk_count",
        "answer_generate_time_ms",
        "answer_total_time_ms",
        "kv_engine_time_to_first_token_ms",
        "answer_time_to_first_token_ms",
        "query_to_answer_ms",
        "query_to_first_token_ms",
        "kv_store_gpu_mb",
    ]
