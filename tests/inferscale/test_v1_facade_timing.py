from __future__ import annotations

from typing import Any

import pytest
from library_fakes import (
    FakeChunkStore,
    FakeEmbedder,
    FakeEncoder,
    FakeEngine,
    FakeIndex,
    FakeTokenizer,
)

import inferscale.v1.api as api_module
import inferscale.v1.retrieval as retrieval_module
from inferscale.v1 import Chunk, InferScale, InferScaleConfig
from inferscale.v1.serving.vllm import GenerationOutput

EMBED_BATCH_S = 0.75
EMBED_QUERY_S = 0.25
INDEX_SEARCH_S = 0.125
FETCH_S = 0.5
TOKENIZE_QUERY_S = 0.0625
COMPOSE_S = 1.0
REGISTRATION_S = 5.0
GENERATE_S = 2.0
CLEANUP_S = 8.0
ENCODE_CHUNK_S = 1.5
ENCODE_SCAFFOLD_PART_S = 4.0
ENGINE_TTFT_MS = 12.0
INDEX_REPORTED_SEARCH_MS = 0.5
PREPARATION_MS = (FETCH_S + TOKENIZE_QUERY_S + COMPOSE_S) * 1000
RETRIEVAL_MS = (EMBED_QUERY_S + INDEX_SEARCH_S) * 1000
QUESTION = "Who has a cat named Miso?"


class ManualClock:
    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SlowEmbedder(FakeEmbedder):
    def __init__(self, clock: ManualClock) -> None:
        super().__init__()
        self.clock = clock

    def embed_documents(self, texts):
        self.clock.advance(EMBED_BATCH_S)
        return super().embed_documents(texts)

    def embed_query(self, text):
        self.clock.advance(EMBED_QUERY_S)
        return super().embed_query(text)


class SlowIndex(FakeIndex):
    def __init__(self, clock: ManualClock) -> None:
        super().__init__()
        self.clock = clock

    def search(self, query_vector, top_k, filters=None):
        self.clock.advance(INDEX_SEARCH_S)
        return super().search(query_vector, top_k, filters)


class SlowStore(FakeChunkStore):
    def __init__(self, clock: ManualClock) -> None:
        super().__init__()
        self.clock = clock

    def get_chunk_memories(self, chunk_ids):
        self.clock.advance(FETCH_S)
        return super().get_chunk_memories(chunk_ids)


class SlowEncoder(FakeEncoder):
    def __init__(self, clock: ManualClock, **settings: Any) -> None:
        super().__init__(**settings)
        self.clock = clock
        self.encoded_chunks = 0

    def encode(self, chunk_id, token_ids):
        self.clock.advance(ENCODE_SCAFFOLD_PART_S)
        return super().encode(chunk_id, token_ids)

    def encode_plan(self, plan):
        self.clock.advance(ENCODE_CHUNK_S)
        self.encoded_chunks += 1
        return super().encode_plan(plan)

    def compose(self, chunks):
        self.clock.advance(COMPOSE_S)
        return super().compose(chunks)


class SlowTokenizer(FakeTokenizer):
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock

    def apply_chat_template(self, messages, **options):
        self.clock.advance(TOKENIZE_QUERY_S)
        return super().apply_chat_template(messages, **options)


class SlowEngine(FakeEngine):
    def __init__(self, clock: ManualClock) -> None:
        super().__init__()
        self.clock = clock
        self.tokenizer_instance = SlowTokenizer(clock)

    def generate(self, prompt_token_ids, *, max_tokens, temperature, top_p) -> GenerationOutput:
        started_at = self.clock.perf_counter()
        self.clock.advance(GENERATE_S)
        return GenerationOutput(
            text="answer",
            ttft_ms=ENGINE_TTFT_MS,
            started_at=started_at,
            finished_at=self.clock.perf_counter(),
        )


def taking(clock: ManualClock, seconds: float, function):
    def slowed(*args, **kwargs):
        clock.advance(seconds)
        return function(*args, **kwargs)

    return slowed


@pytest.fixture
def clock(monkeypatch) -> ManualClock:
    clock = ManualClock()
    monkeypatch.setattr(api_module, "time", clock)
    monkeypatch.setattr(retrieval_module, "time", clock)
    monkeypatch.setattr(
        api_module,
        "register_user_memory",
        taking(clock, REGISTRATION_S, api_module.register_user_memory),
    )
    monkeypatch.setattr(
        api_module, "remove_user_memory", taking(clock, CLEANUP_S, api_module.remove_user_memory)
    )
    monkeypatch.setattr(
        api_module, "ChunkedRopeEncoder", lambda **settings: SlowEncoder(clock, **settings)
    )
    return clock


@pytest.fixture
def facade(clock):
    config = InferScaleConfig.from_dict(
        {"model": "fake-model", "top_k": 2, "context_window": 1, "engine": {"block_size": 4}}
    )
    facade = InferScale(
        config,
        embedder=SlowEmbedder(clock),
        index=SlowIndex(clock),
        chunk_store=SlowStore(clock),
        engine=SlowEngine(clock),
    )
    yield facade
    facade.close()


@pytest.fixture
def serving_facade(facade):
    facade.precompute(
        [
            Chunk("c1", "Alice has a cat named Miso."),
            Chunk("c2", "Bob teaches chemistry."),
            Chunk("c3", "Alice lives in Berlin."),
        ]
    )
    facade.start()
    return facade


def timings(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: value for name, value in metrics.items() if name.endswith("_ms")}


def test_registration_time_is_reported_but_excluded_from_every_latency(serving_facade) -> None:
    result = serving_facade.query(QUESTION)

    assert timings(result.metrics) == {
        "kv_fetch_time_ms": FETCH_S * 1000,
        "kv_compose_time_ms": COMPOSE_S * 1000,
        "kv_store_write_time_ms": REGISTRATION_S * 1000,
        "answer_generate_time_ms": GENERATE_S * 1000,
        "answer_total_time_ms": PREPARATION_MS + GENERATE_S * 1000,
        "kv_engine_time_to_first_token_ms": ENGINE_TTFT_MS,
        "answer_time_to_first_token_ms": PREPARATION_MS + ENGINE_TTFT_MS,
        "query_to_answer_ms": RETRIEVAL_MS + PREPARATION_MS + GENERATE_S * 1000,
        "query_to_first_token_ms": RETRIEVAL_MS + PREPARATION_MS + ENGINE_TTFT_MS,
        "query_embedding_time_ms": EMBED_QUERY_S * 1000,
        "vector_db_query_time_ms": INDEX_REPORTED_SEARCH_MS,
        "query_retrieval_time_ms": EMBED_QUERY_S * 1000 + INDEX_REPORTED_SEARCH_MS,
    }
    assert result.ttft_ms == ENGINE_TTFT_MS


def test_precompute_times_embedding_and_chunk_encoding_separately(facade) -> None:
    stats = facade.precompute(
        [Chunk("c1", "first"), Chunk("c2", "second"), Chunk("c3", "third", [7, 8])]
    )

    assert stats.embed_time_ms == EMBED_BATCH_S * 1000
    assert stats.encode_time_ms == 3 * ENCODE_CHUNK_S * 1000


def test_allocator_fragments_are_released_after_every_25_chunks_of_a_batch(
    facade, monkeypatch
) -> None:
    encoded_before_release = []

    with monkeypatch.context() as during_precompute:
        during_precompute.setattr(
            api_module,
            "empty_cuda_cache",
            lambda: encoded_before_release.append(facade._encoder.encoded_chunks),
        )
        facade.precompute([Chunk(f"a{number}", "text", [number + 1]) for number in range(60)])
        facade.precompute([Chunk(f"b{number}", "text", [number + 1]) for number in range(24)])
        facade.precompute([Chunk(f"c{number}", "text", [number + 1]) for number in range(25)])

    assert encoded_before_release == [25, 50, 60 + 24 + 25]
