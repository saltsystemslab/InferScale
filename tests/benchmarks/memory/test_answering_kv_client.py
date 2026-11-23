from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from library_fakes import FakeChunkStore, FakeTensor, FakeTokenizer
from loguru import logger
from memory_config import make_memory_config, memory_runtime

from benchmarks.memory import answer_kv, composer, sample_precompute
from benchmarks.memory.answer_kv import KVAnswerClient
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, QuestionAnswer, Turn
from benchmarks.memory.mem0.fact_catalog import MemoryFact, fact_catalog_hits
from benchmarks.memory.prompting import (
    build_memory_prompt_token_ids,
    build_memory_query,
    extract_memory_scaffold_token_ids,
)
from inferscale.v1.kv.registry import (
    get_gpu_memory_store,
    namespace_stats,
    register_user_memory,
)
from inferscale.v1.serving.vllm import GenerationOutput
from inferscale.v1.types import EncodingPlan, KVChunk, SearchHit

torch = pytest.importorskip("torch")

LAYERS = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")
TOKENIZE_MS = 1
ENCODE_CHUNK_MS = 10
FETCH_CHUNKS_MS = 2
COMPOSE_MS = 5
REGISTER_LAYER_MS = 3
STORE_STATISTICS_MS = 4
GENERATE_MS = 40
ENGINE_TTFT_MS = 12.0
SCAFFOLD_CHUNKS = 3


class Clock:
    def __init__(self) -> None:
        self.now_s = 100.0

    def perf_counter(self) -> float:
        return self.now_s

    def advance_ms(self, milliseconds: float) -> None:
        self.now_s += milliseconds / 1000


class TimedTokenizer(FakeTokenizer):
    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.clock.advance_ms(TOKENIZE_MS)
        return super().encode(text, add_special_tokens)


class ComposedTensor(FakeTensor):
    def __init__(self, array: Any, clock: Clock, device: str = "cpu") -> None:
        super().__init__(array, device)
        self.clock = clock

    def to(self, device: Any = None, **kwargs: Any) -> ComposedTensor:
        self.clock.advance_ms(REGISTER_LAYER_MS)
        target = device if device is not None else kwargs.get("device", self.device)
        return ComposedTensor(self.array, self.clock, str(target))


class TensorEncoder:
    def __init__(self, clock: Clock, *, tokenizer: Any, cos_table: Any, sin_table: Any) -> None:
        self.clock = clock
        self.device = "cpu"
        self.max_position = 4096
        self.tokenizer = tokenizer
        self.cos_table = cos_table
        self.sin_table = sin_table
        self.loads_model_weights = True
        self.released_model = False
        self.closed = False
        self.composed: list[list[str]] = []

    def encode(self, chunk_id: str, token_ids: list[int]) -> KVChunk:
        self.clock.advance_ms(ENCODE_CHUNK_MS)
        return KVChunk(chunk_id=chunk_id, token_ids=list(token_ids), kv_by_layer=_kv(token_ids))

    def encode_plan(self, plan: EncodingPlan) -> KVChunk:
        self.clock.advance_ms(ENCODE_CHUNK_MS)
        return KVChunk(
            chunk_id=plan.chunk_id,
            token_ids=list(plan.target_token_ids),
            kv_by_layer=_kv(plan.target_token_ids),
            context_ids=plan.context_ids,
            context_prefix_tokens=len(plan.context_token_ids),
        )

    def compose(self, chunks: list[KVChunk]) -> dict[str, Any]:
        self.clock.advance_ms(COMPOSE_MS)
        self.composed.append([chunk.chunk_id for chunk in chunks])
        return {
            layer: ComposedTensor(
                torch.cat([chunk.kv_by_layer[layer] for chunk in chunks], dim=1).numpy(),
                self.clock,
            )
            for layer in LAYERS
        }

    def release_model(self) -> None:
        self.released_model = True

    def close(self) -> None:
        self.closed = True


def _kv(token_ids: list[int]) -> dict[str, Any]:
    tokens = torch.tensor(token_ids, dtype=torch.float32).reshape(1, -1, 1, 1)
    return {layer: tokens.repeat(2, 1, 1, 1) for layer in LAYERS}


class ChunkStore(FakeChunkStore):
    def __init__(self, clock: Clock) -> None:
        super().__init__()
        self.clock = clock
        self.totals: dict[str, float] = {}
        self.transfers_per_fetch: list[dict[str, float]] = []
        self.host_mb: float | None = None
        self.released: list[str] = []

    def get_chunk_memories(self, chunk_ids: Any) -> Any:
        self.clock.advance_ms(FETCH_CHUNKS_MS)
        if self.transfers_per_fetch:
            for name, amount in self.transfers_per_fetch.pop(0).items():
                self.totals[name] = self.totals.get(name, 0) + amount
        return super().get_chunk_memories(chunk_ids)

    def release_staging(self, user_id: str) -> None:
        self.released.append(user_id)

    def transfer_totals(self) -> dict[str, float]:
        return dict(self.totals)

    def get_stats(self) -> dict[str, Any]:
        self.clock.advance_ms(STORE_STATISTICS_MS)
        stats = super().get_stats()
        if self.host_mb is not None:
            stats["total_host_mb"] = self.host_mb
        return stats


class Engine:
    def __init__(self, clock: Clock, namespace_of: Callable[[], str]) -> None:
        self.clock = clock
        self.namespace_of = namespace_of
        self.tokenizer_instance = TimedTokenizer(clock)
        self.started = False
        self.started_with: dict[str, Any] | None = None
        self.prompts: list[list[int]] = []
        self.sampling: list[tuple[int, float, float]] = []
        self.memories_seen_while_generating: list[dict[str, Any]] = []

    @property
    def tokenizer(self) -> Any:
        return self.tokenizer_instance

    def start(self, *, kv_transfer_config: Any = None) -> float:
        self.started = True
        self.started_with = dict(kv_transfer_config)
        return 0.0

    def generate(
        self, prompt_token_ids: Any, *, max_tokens: int, temperature: float, top_p: float
    ) -> GenerationOutput:
        self.prompts.append(list(prompt_token_ids))
        self.sampling.append((max_tokens, temperature, top_p))
        store = get_gpu_memory_store(self.namespace_of())
        self.memories_seen_while_generating.append(
            {
                user_id: list(store.get_user_memory(user_id).token_ids)
                for user_id in store.get_all_user_ids()
            }
        )
        started_at = self.clock.perf_counter()
        self.clock.advance_ms(GENERATE_MS)
        return GenerationOutput(
            text="tea",
            ttft_ms=ENGINE_TTFT_MS,
            started_at=started_at,
            finished_at=self.clock.perf_counter(),
        )

    def close(self) -> None:
        self.started = False


class Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.clock = Clock()
        self.tmp_path = tmp_path
        self.clients: list[KVAnswerClient] = []
        self.engines: list[Engine] = []
        self.encoders: list[TensorEncoder] = []
        self.stores: list[ChunkStore] = []
        self.store_requests: list[tuple[str, dict[str, Any]]] = []
        self.loads: list[dict[str, Any]] = []
        self.saves: list[dict[str, Any]] = []
        self.save_failure: BaseException | None = None
        self.standalone_tokenizer_loads: list[str] = []
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        monkeypatch.setattr(answer_kv, "time", self.clock)
        monkeypatch.setattr(composer, "time", self.clock)
        monkeypatch.setattr(answer_kv, "VLLMEngine", self._engine)
        monkeypatch.setattr(sample_precompute, "ChunkedRopeEncoder", self)
        monkeypatch.setattr(sample_precompute, "build_chunk_store", self._chunk_store)
        monkeypatch.setattr(sample_precompute, "load_encoder_tokenizer", self._standalone_tokenizer)
        real_load = sample_precompute.load_sample_chunks
        real_save = sample_precompute.save_sample_chunks

        def load(path: Path, **options: Any) -> Any:
            self.loads.append({"path": path, **options})
            return real_load(path, **options)

        def save(path: Path, **options: Any) -> None:
            self.saves.append(
                {
                    "path": path,
                    "facts_with_tensors": sorted(
                        fact_id
                        for fact_id, chunk in options["fact_chunks"].items()
                        if chunk.kv_by_layer
                    ),
                }
            )
            if self.save_failure is not None:
                raise self.save_failure
            real_save(path, **options)

        monkeypatch.setattr(sample_precompute, "load_sample_chunks", load)
        monkeypatch.setattr(sample_precompute, "save_sample_chunks", save)

    def __call__(self, *, model: str, dtype: str, device: str, max_position: int) -> TensorEncoder:
        encoder = TensorEncoder(
            self.clock,
            tokenizer=TimedTokenizer(self.clock),
            cos_table=torch.ones(4, 2),
            sin_table=torch.zeros(4, 2),
        )
        self.encoders.append(encoder)
        return encoder

    def from_tables(
        self,
        *,
        model: str,
        device: str,
        max_position: int,
        tokenizer: Any,
        cos_table: Any,
        sin_table: Any,
    ) -> TensorEncoder:
        encoder = TensorEncoder(
            self.clock, tokenizer=tokenizer, cos_table=cos_table, sin_table=sin_table
        )
        encoder.loads_model_weights = False
        self.encoders.append(encoder)
        return encoder

    def _engine(self, model: str, dtype: str, engine_config: Any) -> Engine:
        engine = Engine(self.clock, lambda: self.clients[-1].namespace)
        self.engines.append(engine)
        return engine

    def _chunk_store(self, backend: str, **options: Any) -> ChunkStore:
        self.store_requests.append((backend, options))
        store = ChunkStore(self.clock)
        self.stores.append(store)
        return store

    def _standalone_tokenizer(self, model: str) -> TimedTokenizer:
        self.standalone_tokenizer_loads.append(model)
        return TimedTokenizer(self.clock)

    def config(self, **values: Any) -> MemoryRunConfig:
        settings: dict[str, Any] = {
            "runtime": memory_runtime(storage={"runtime_root": str(self.tmp_path)}),
            "run_id": "kv-run",
            "skip_judge": True,
            "kv_chunk_cache_enabled": False,
            "inferscale": _library_section(),
        }
        settings.update(values)
        return make_memory_config(**settings)

    def client(self, **values: Any) -> KVAnswerClient:
        client = KVAnswerClient(self.config(**values))
        self.clients.append(client)
        return client

    def serving_client(self, sample: ConversationSample) -> KVAnswerClient:
        client = self.client()
        client.precompute_sample_cache(sample, _catalog_hits())
        client.start_llm()
        client.prepare_sample(sample)
        return client

    def cache_files(self) -> list[Path]:
        return sorted(self.tmp_path.rglob("*.pt"))

    def close(self) -> None:
        for client in self.clients:
            client.close()


def _library_section(**kv: Any) -> dict[str, Any]:
    return {
        "engine": {"block_size": 16},
        "kv": {"device": "cpu", "max_position": 4096, **kv},
    }


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Harness]:
    harness = Harness(monkeypatch, tmp_path)
    yield harness
    harness.close()


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink)


def _sample(sample_id: str = "sample-1") -> ConversationSample:
    turns = [
        Turn(
            sample_id=sample_id,
            session_id="session_1",
            session_index=1,
            turn_index=turn_index,
            speaker="Alice",
            text=f"I like drink {turn_index}.",
            timestamp="2026-01-02",
        )
        for turn_index in range(3)
    ]
    questions = [
        QuestionAnswer(
            sample_id=sample_id,
            question_id=f"q{index}",
            question=f"Question {index}?",
            answer="tea",
            category="1",
        )
        for index in range(2)
    ]
    return ConversationSample(sample_id=sample_id, turns=turns, qa=questions, raw={})


def _fact(index: int) -> MemoryFact:
    return MemoryFact(
        id=f"fact-{index}",
        text=f"Alice likes drink {index}.",
        created_at="2026-01-02T00:00:00+00:00",
        timestamp_epoch=1767312000,
        sample_id="sample-1",
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=index,
        source_turn_id=f"sample-1:session_1:{index}",
        speaker="Alice",
        role="user",
    )


def _catalog_hits() -> list[SearchHit]:
    return fact_catalog_hits([_fact(0), _fact(1), _fact(2)])


def _tokenizing_ms(tokenize: Callable[[TimedTokenizer], Any]) -> float:
    clock = Clock()
    started = clock.perf_counter()
    tokenize(TimedTokenizer(clock))
    return (clock.perf_counter() - started) * 1000


def _query_building_ms(sample: ConversationSample) -> float:
    return _tokenizing_ms(
        lambda tokenizer: build_memory_query(
            tokenizer,
            sample.qa[0],
            block_size=16,
            max_position=4096,
            max_model_len=32768,
            max_answer_tokens=16,
        )
    )


def _answer(client: KVAnswerClient, sample: ConversationSample, hits: list[SearchHit], **options):
    return client.answer_with_retrieved_memory(
        sample=sample,
        qa=sample.qa[0],
        hits=hits,
        **{"max_tokens": 16, "temperature": 0.0, "top_p": 1.0, **options},
    )


def test_precompute_must_finish_before_the_engine_starts(harness: Harness) -> None:
    sample = _sample()
    client = harness.client()

    with pytest.raises(RuntimeError, match="At least one sample must be precomputed"):
        client.start_llm()
    assert harness.engines[0].started is False

    client.precompute_sample_cache(sample, _catalog_hits())
    client.start_llm()
    client.start_llm()

    assert harness.engines[0].started is True
    assert harness.encoders[0].released_model is True
    with pytest.raises(RuntimeError, match="KV precompute must finish for every sample"):
        client.precompute_sample_cache(sample, _catalog_hits())


def test_connector_reads_composed_memory_from_a_gpu_store_whatever_the_chunk_store_backend(
    harness: Harness,
) -> None:
    client = harness.client(inferscale=_library_section(store_backend="cpu", staging_slots=9))
    metrics = client.precompute_sample_cache(_sample(), _catalog_hits())
    client.start_llm()

    assert harness.engines[0].started_with == {
        "kv_connector": "MemoryKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "inferscale.v1.kv.connector",
        "kv_connector_extra_config": {
            "memory_namespace": client.namespace,
            "log_memory_hits": True,
            "memory_store_backend": "gpu",
            "num_staging_slots": 4,
            "default_user_id": client.active_user_id,
        },
    }
    assert client.namespace.startswith("kv-run-")
    assert client.active_user_id == f"{client.namespace}-active"
    with pytest.raises(RuntimeError, match="already exists with backend 'gpu'; requested 'cpu'"):
        get_gpu_memory_store(client.namespace, backend="cpu")
    assert harness.store_requests == [("cpu", {"device": "cpu", "top_k": 50, "staging_slots": 9})]
    assert metrics["kv_chunk_cache_residency_is_gpu"] == 0


def test_answer_hands_the_composed_memory_to_the_connector_only_while_generating(
    harness: Harness,
) -> None:
    sample = _sample()
    client = harness.serving_client(sample)
    hits = _catalog_hits()[:2]

    result = _answer(client, sample, hits, max_tokens=64, temperature=0.5, top_p=0.9)

    tokenizer = FakeTokenizer()
    query = build_memory_query(
        tokenizer,
        sample.qa[0],
        block_size=16,
        max_position=4096,
        max_model_len=32768,
        max_answer_tokens=64,
    )
    memory = build_memory_prompt_token_ids(
        tokenizer,
        sample,
        hits,
        memory_token_budget=query.memory_token_budget,
        memory_scaffold=query.scaffold,
    )
    engine = harness.engines[0]
    assert engine.prompts == [[*memory.token_ids, *query.query_tokens.token_ids]]
    assert engine.sampling == [(64, 0.5, 0.9)]
    assert engine.memories_seen_while_generating == [{client.active_user_id: memory.token_ids}]
    assert namespace_stats(client.namespace)["num_users"] == 0
    assert harness.encoders[0].composed == [["__header__", "fact-1", "fact-0", "__footer__"]]
    assert harness.stores[0].lookup_batches == [["fact-1", "fact-0"]]
    assert harness.stores[0].released == ["fact-1", "fact-0"]
    assert result.content == "tea"
    assert result.ttft_ms == ENGINE_TTFT_MS
    metrics = result.metrics
    assert metrics["kv_memory_tokens"] == len(memory.token_ids)
    assert metrics["kv_query_tokens"] == len(query.query_tokens.token_ids)
    assert metrics["kv_query_bos_stripped"] == 0
    assert metrics["kv_selected_fact_ids"] == ["fact-1", "fact-0"]
    assert metrics["memory_retrieved_fact_ids"] == ["fact-0", "fact-1"]
    assert metrics["kv_context_window"] == 0
    assert metrics["kv_block_size"] == 16
    assert metrics["kv_prefix_caching"] == 1
    assert metrics["kv_loaded_memory_tokens"] == len(memory.token_ids) // 16 * 16
    assert metrics["kv_recomputed_memory_tail_tokens"] == len(memory.token_ids) % 16
    assert metrics["kv_fact_tokens_end"] == len(memory.token_ids) - len(
        query.scaffold.footer_token_ids
    )
    assert metrics["memory_token_budget"] == query.memory_token_budget
    assert metrics["kv_store_gpu_mb"] > 0
    assert metrics["kv_precomputed_chunks"] == 3
    assert metrics["kv_chunk_cache_hit"] == 0


class RetokenizingTokenizer(TimedTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [token_id + 1 for token_id in super().encode(text, add_special_tokens)]


def test_precomputed_tokens_that_differ_from_the_engine_tokenizer_are_never_served(
    harness: Harness,
) -> None:
    sample = _sample()
    client = harness.client()
    client.precompute_sample_cache(sample, _catalog_hits())
    harness.engines[0].tokenizer_instance = RetokenizingTokenizer(harness.clock)
    client.start_llm()
    client.prepare_sample(sample)

    with pytest.raises(RuntimeError, match="Precomputed memory tokens differ from the live"):
        _answer(client, sample, _catalog_hits())

    assert harness.engines[0].prompts == []
    assert namespace_stats(client.namespace)["num_users"] == 0


def test_precompute_time_covers_encoding_and_its_statistics(harness: Harness) -> None:
    facts = _catalog_hits()
    client = harness.client()

    metrics = client.precompute_sample_cache(_sample(), facts)

    tokenizing_ms = len(facts) * TOKENIZE_MS + _tokenizing_ms(
        lambda tokenizer: extract_memory_scaffold_token_ids(tokenizer, block_size=16)
    )
    encoding_ms = (len(facts) + SCAFFOLD_CHUNKS) * ENCODE_CHUNK_MS
    assert metrics["kv_precompute_time_ms"] == pytest.approx(
        tokenizing_ms + encoding_ms + STORE_STATISTICS_MS
    )
    assert metrics["kv_chunk_cache_residency_is_gpu"] == 1
    assert metrics["kv_chunk_cache_hit"] == 0
    assert metrics["kv_chunk_cache_residency"] == "gpu"
    assert metrics["kv_precomputed_chunks"] == 3
    assert metrics["kv_precomputed_chunks_with_prefix"] == 6
    assert metrics["kv_precomputed_layers"] == 2


def test_answer_times_composition_verification_and_registration_separately(
    harness: Harness,
) -> None:
    sample = _sample()
    client = harness.serving_client(sample)
    hits = _catalog_hits()[:2]

    metrics = _answer(client, sample, hits).metrics

    scaffold = extract_memory_scaffold_token_ids(FakeTokenizer(), block_size=16)
    compose_ms = FETCH_CHUNKS_MS + COMPOSE_MS
    verify_ms = _tokenizing_ms(
        lambda tokenizer: build_memory_prompt_token_ids(
            tokenizer, sample, hits, memory_scaffold=scaffold
        )
    )
    register_ms = len(LAYERS) * REGISTER_LAYER_MS
    before_generation_ms = _query_building_ms(sample) + compose_ms + verify_ms + register_ms
    bookkeeping_ms = verify_ms + register_ms
    assert verify_ms > 0
    assert metrics["kv_compose_time_ms"] == pytest.approx(compose_ms)
    assert metrics["kv_verify_time_ms"] == pytest.approx(verify_ms)
    assert metrics["kv_store_write_time_ms"] == pytest.approx(register_ms)
    assert metrics["answer_generate_time_ms"] == pytest.approx(GENERATE_MS)
    assert metrics["kv_engine_time_to_first_token_ms"] == ENGINE_TTFT_MS
    assert metrics["answer_time_to_first_token_ms"] == pytest.approx(
        before_generation_ms - bookkeeping_ms + ENGINE_TTFT_MS
    )
    assert metrics["answer_total_time_ms"] == pytest.approx(
        before_generation_ms - bookkeeping_ms + GENERATE_MS
    )
    assert "query_to_answer_ms" not in metrics
    assert "query_to_first_token_ms" not in metrics


def test_latency_is_measured_from_the_start_times_the_caller_passes(harness: Harness) -> None:
    sample = _sample()
    client = harness.serving_client(sample)
    hits = _catalog_hits()[:2]
    retrieval_ms = 100
    ttft_started_at = harness.clock.perf_counter()
    query_started_at = ttft_started_at - retrieval_ms / 1000
    harness.clock.advance_ms(30)

    metrics = _answer(
        client, sample, hits, ttft_started_at=ttft_started_at, query_started_at=query_started_at
    ).metrics

    compose_ms = FETCH_CHUNKS_MS + COMPOSE_MS
    after_request_start_ms = 30 + _query_building_ms(sample) + compose_ms
    assert metrics["answer_time_to_first_token_ms"] == pytest.approx(
        after_request_start_ms + ENGINE_TTFT_MS
    )
    assert metrics["answer_total_time_ms"] == pytest.approx(after_request_start_ms + GENERATE_MS)
    assert metrics["query_to_first_token_ms"] == pytest.approx(
        retrieval_ms + after_request_start_ms + ENGINE_TTFT_MS
    )
    assert metrics["query_to_answer_ms"] == pytest.approx(
        retrieval_ms + after_request_start_ms + GENERATE_MS
    )


def test_transfer_metrics_are_the_growth_of_the_store_totals_during_composition(
    harness: Harness,
) -> None:
    sample = _sample()
    client = harness.serving_client(sample)
    store = harness.stores[0]
    store.host_mb = 2.5
    store.transfers_per_fetch = [
        {
            "total_h2d_latency_ms": 2.0,
            "total_staging_stall_ms": 0.5,
            "total_bytes_transferred": 300,
        },
        {"total_staging_stall_ms": 3.0},
        {"total_h2d_latency_ms": 1.0, "total_staging_stall_ms": 5.0, "total_bytes_transferred": 64},
    ]

    overlapped = _answer(client, sample, _catalog_hits()).metrics
    only_stalled = _answer(client, sample, _catalog_hits()).metrics
    mostly_stalled = _answer(client, sample, _catalog_hits()).metrics
    idle = _answer(client, sample, _catalog_hits()).metrics

    assert overlapped["kv_h2d_latency_ms"] == 2.0
    assert overlapped["kv_staging_stall_ms"] == 0.5
    assert overlapped["kv_h2d_bytes"] == 300
    assert overlapped["kv_h2d_overlap_ratio"] == 0.75
    assert overlapped["kv_store_host_mb"] == 2.5
    assert only_stalled["kv_h2d_latency_ms"] == 0.0
    assert only_stalled["kv_staging_stall_ms"] == 3.0
    assert only_stalled["kv_h2d_bytes"] == 0
    assert only_stalled["kv_h2d_overlap_ratio"] == 0.0
    assert mostly_stalled["kv_h2d_latency_ms"] == 1.0
    assert mostly_stalled["kv_staging_stall_ms"] == 5.0
    assert mostly_stalled["kv_h2d_bytes"] == 64
    assert mostly_stalled["kv_h2d_overlap_ratio"] == 0.0
    assert idle["kv_h2d_latency_ms"] == 0.0
    assert idle["kv_staging_stall_ms"] == 0.0
    assert idle["kv_h2d_bytes"] == 0
    assert idle["kv_h2d_overlap_ratio"] == 0.0


def test_preparing_the_next_sample_releases_the_previous_one(harness: Harness) -> None:
    first, second = _sample(), _sample("sample-2")
    client = harness.client()
    client.precompute_sample_cache(first, _catalog_hits())
    client.precompute_sample_cache(second, _catalog_hits()[:1])
    client.start_llm()
    first_store, second_store = harness.stores

    client.prepare_sample(first)
    client.prepare_sample(first)
    assert first_store.get_all_user_ids() == ["fact-0", "fact-1", "fact-2"]
    _leave_memory_in_the_connector_store(client)

    client.prepare_sample(second)

    assert first_store.get_all_user_ids() == []
    assert second_store.get_all_user_ids() == ["fact-0"]
    assert namespace_stats(client.namespace)["num_users"] == 0
    assert _answer(client, second, _catalog_hits()[:1]).metrics["kv_precomputed_chunks"] == 1
    with pytest.raises(RuntimeError, match="KV cache for sample_id=sample-1 was not precomputed"):
        client.prepare_sample(first)


def _leave_memory_in_the_connector_store(client: KVAnswerClient) -> None:
    register_user_memory(
        client.namespace, user_id="left over", kv_by_layer={}, num_tokens=0, token_ids=[]
    )
    assert namespace_stats(client.namespace)["num_users"] == 1


def test_chunk_cache_is_written_on_a_miss_and_read_by_the_next_run(harness: Harness) -> None:
    sample = _sample()
    hits = _catalog_hits()

    first = harness.client(kv_chunk_cache_enabled=True)
    missed = first.precompute_sample_cache(sample, hits)
    first.start_llm()
    first.prepare_sample(sample)
    encoded_answer = _answer(first, sample, hits)
    first.close()

    second = harness.client(kv_chunk_cache_enabled=True)
    hit = second.precompute_sample_cache(sample, hits)
    second.start_llm()
    second.prepare_sample(sample)
    cached_answer = _answer(second, sample, hits)

    [cache_file] = harness.cache_files()
    assert cache_file.parent == (
        harness.tmp_path
        / ".cache/kv-chunks/v2/meta-llama__Llama-3.1-8B-Instruct/bfloat16-w0-p4096-b16"
    )
    assert (missed["kv_chunk_cache_hit"], hit["kv_chunk_cache_hit"]) == (0, 1)
    assert [save["path"] for save in harness.saves] == [cache_file]
    assert [load["path"] for load in harness.loads] == [cache_file, cache_file]
    encoding_encoder, cached_encoder = harness.encoders
    assert encoding_encoder.loads_model_weights is True
    assert cached_encoder.loads_model_weights is False
    assert harness.standalone_tokenizer_loads == [second.config.model]
    assert harness.engines[1].prompts == harness.engines[0].prompts
    assert cached_encoder.composed == [["header", "fact-2", "fact-1", "fact-0", "footer"]]
    for name in (
        "kv_memory_tokens",
        "kv_selected_fact_ids",
        "kv_loaded_memory_tokens",
        "kv_precomputed_tokens",
        "kv_precomputed_gpu_mb",
        "llama_kv_chunk_tensor_gpu_bytes",
        "llama_kv_prefix_tensor_gpu_bytes",
    ):
        assert cached_answer.metrics[name] == encoded_answer.metrics[name]


def test_chunk_cache_is_saved_while_the_composer_still_holds_the_tensors(
    harness: Harness,
) -> None:
    client = harness.client(kv_chunk_cache_enabled=True)

    client.precompute_sample_cache(_sample(), _catalog_hits())

    assert [save["facts_with_tensors"] for save in harness.saves] == [
        ["fact-0", "fact-1", "fact-2"]
    ]
    assert harness.stores[0].get_all_user_ids() == ["fact-0", "fact-1", "fact-2"]


def test_failed_chunk_cache_save_does_not_fail_the_precompute(
    harness: Harness, warnings: list[str]
) -> None:
    sample = _sample()
    harness.save_failure = OSError("No space left on device")
    client = harness.client(kv_chunk_cache_enabled=True)

    metrics = client.precompute_sample_cache(sample, _catalog_hits())
    client.start_llm()
    client.prepare_sample(sample)

    assert metrics["kv_precomputed_chunks"] == 3
    assert _answer(client, sample, _catalog_hits()).content == "tea"
    assert harness.cache_files() == []
    [warning] = warnings
    assert warning.startswith("Failed to save KV chunk cache ")
    assert warning.endswith(".pt: No space left on device")


@pytest.mark.parametrize("store_backend, fact_device", [("gpu", "cuda:1"), ("cpu", "cpu")])
def test_cached_fact_chunks_load_to_host_memory_only_under_the_cpu_store(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, store_backend: str, fact_device: str
) -> None:
    monkeypatch.setattr(sample_precompute, "load_sample_chunks", _recording_miss(harness.loads))
    client = harness.client(
        kv_chunk_cache_enabled=True,
        inferscale={
            "engine": {"block_size": 16},
            "kv": {"device": "cuda:1", "max_position": 4096, "store_backend": store_backend},
        },
    )

    client.precompute_sample_cache(_sample(), _catalog_hits())

    [load] = harness.loads
    assert (load["device"], load["scaffold_device"]) == (fact_device, "cuda:1")
    assert load["expected_fact_ids"] == ["fact-0", "fact-1", "fact-2"]
    assert load["expected_meta"]["model"] == client.config.model
    assert harness.store_requests == [
        (store_backend, {"device": "cuda:1", "top_k": 50, "staging_slots": 4})
    ]


def _recording_miss(loads: list[dict[str, Any]]) -> Callable[..., None]:
    def load(path: Path, **options: Any) -> None:
        loads.append({"path": path, **options})

    return load
