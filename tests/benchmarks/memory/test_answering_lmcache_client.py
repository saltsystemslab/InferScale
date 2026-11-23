from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from library_fakes import FakeTokenizer
from memory_config import make_memory_config, memory_config_data

from benchmarks.common.lmcache.runtime import WarmupStats
from benchmarks.common.lmcache.segments import contains_separator, separator_token_ids
from benchmarks.memory import answer_lmcache
from benchmarks.memory.answer_lmcache import LMCacheBlendAnswerClient
from benchmarks.memory.data import ConversationSample, QuestionAnswer
from benchmarks.memory.mem0.fact_catalog import MemoryFact, fact_catalog_hits
from benchmarks.memory.prompting import (
    build_memory_query,
    extract_memory_scaffold_token_ids,
    select_memory_facts,
)
from inferscale.v1.serving.vllm import GenerationOutput
from inferscale.v1.types import SearchHit

SEPARATOR = " # # "
TOKENIZE_MS = 1
RESET_LOAD_STATS_MS = 3
SNAPSHOT_LOAD_STATS_MS = 7
GENERATE_MS = 40
ENGINE_TTFT_MS = 12.0


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


class Engine:
    def __init__(self, clock: Clock, events: list[str]) -> None:
        self.clock = clock
        self.events = events
        self.tokenizer = TimedTokenizer(clock)
        self.llm = object()
        self.started = False

    def start(self, *, kv_transfer_config: Any = None) -> float:
        self.events.append("engine_start")
        self.started = True
        return 0.0

    def generate(self, prompt_token_ids: Any, **sampling: Any) -> GenerationOutput:
        self.events.append("generate")
        started_at = self.clock.perf_counter()
        self.clock.advance_ms(GENERATE_MS)
        return GenerationOutput(
            text="tea",
            ttft_ms=ENGINE_TTFT_MS,
            started_at=started_at,
            finished_at=self.clock.perf_counter(),
        )


class Runtime:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.clock = Clock()
        self.events: list[str] = []
        self.engines: list[Engine] = []
        self.registered_models: list[Any] = []
        self.warmed_batches: list[tuple[list[list[int]], int]] = []
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        monkeypatch.setattr(answer_lmcache, "time", self.clock)
        monkeypatch.setattr(answer_lmcache, "VLLMEngine", self._engine)
        monkeypatch.setattr(answer_lmcache, "register_model", self._register_model)
        monkeypatch.setattr(answer_lmcache, "lmcache_separator", self._separator)
        monkeypatch.setattr(answer_lmcache, "kv_bytes_per_token", lambda: 1000)
        monkeypatch.setattr(answer_lmcache, "clear_cache", self._event("clear_cache"))
        monkeypatch.setattr(answer_lmcache, "require_cache_capacity", lambda **_: 0)
        monkeypatch.setattr(answer_lmcache, "warm_prompts", self._warm)
        monkeypatch.setattr(answer_lmcache, "warmup_metrics", lambda stats: {"warmed": True})
        monkeypatch.setattr(
            answer_lmcache, "reset_load_stats", self._event("reset", RESET_LOAD_STATS_MS)
        )
        monkeypatch.setattr(answer_lmcache, "snapshot_load_stats", self._snapshot)

    def _engine(self, model: str, dtype: str, engine_config: Any) -> Engine:
        engine = Engine(self.clock, self.events)
        self.engines.append(engine)
        return engine

    def _event(self, name: str, elapsed_ms: float = 0.0) -> Any:
        def call(*args: Any) -> None:
            self.clock.advance_ms(elapsed_ms)
            self.events.append(name)

        return call

    def _register_model(self, llm: Any) -> None:
        self.events.append("register_model")
        self.registered_models.append(llm)

    def _separator(self) -> list[int]:
        return separator_token_ids(FakeTokenizer(), SEPARATOR)

    def _warm(self, llm: Any, prompts: Any, *, batch_size: int) -> WarmupStats:
        self.warmed_batches.append(([list(prompt) for prompt in prompts], batch_size))
        return WarmupStats(len(prompts), sum(len(prompt) for prompt in prompts), 7.0)

    def _snapshot(self) -> dict[str, Any]:
        self.clock.advance_ms(SNAPSHOT_LOAD_STATS_MS)
        self.events.append("snapshot")
        return {"tokens_matched": 10_000, "load_time_ms": 3.0}


@pytest.fixture
def lmcache(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    return Runtime(monkeypatch)


def _client(**values: Any) -> LMCacheBlendAnswerClient:
    config = make_memory_config(
        answer_backend="lmcache-blend", vector_backend="qdrant", skip_judge=True, **values
    )
    return LMCacheBlendAnswerClient(config)


def _serving_client(sample: ConversationSample, **values: Any) -> LMCacheBlendAnswerClient:
    client = _client(**values)
    client.start_llm()
    client.prepare_sample(sample)
    return client


def _sample() -> ConversationSample:
    question = QuestionAnswer(
        sample_id="sample-1",
        question_id="q0",
        question="What does Alice like?",
        answer="tea",
        category="1",
    )
    return ConversationSample(sample_id="sample-1", turns=[], qa=[question], raw={})


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


def _hits(count: int = 2) -> list[SearchHit]:
    return fact_catalog_hits([_fact(index) for index in range(count)])


def _tokenizing_ms(tokenize: Callable[[TimedTokenizer], Any]) -> float:
    clock = Clock()
    started = clock.perf_counter()
    tokenize(TimedTokenizer(clock))
    return (clock.perf_counter() - started) * 1000


def _prompt_building_ms(sample: ConversationSample, hits: list[SearchHit]) -> float:
    def build_prompt(tokenizer: TimedTokenizer) -> None:
        build_memory_query(
            tokenizer,
            sample.qa[0],
            block_size=16,
            max_position=32768,
            max_model_len=32768,
            max_answer_tokens=16,
        )
        select_memory_facts(tokenizer, hits)

    return _tokenizing_ms(build_prompt)


def _answer(
    client: LMCacheBlendAnswerClient,
    sample: ConversationSample,
    hits: list[SearchHit],
    **start_times: float,
):
    return client.answer_with_retrieved_memory(
        sample=sample,
        qa=sample.qa[0],
        hits=hits,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        **start_times,
    )


def _kv_section_with_max_position(max_position: int) -> dict[str, Any]:
    library = memory_config_data()["inferscale"]
    return {**library, "kv": {**library["kv"], "max_position": max_position}}


def test_memory_budget_counts_the_separators_between_the_facts(lmcache: Runtime) -> None:
    sample = _sample()
    tokenizer = FakeTokenizer()
    scaffold = extract_memory_scaffold_token_ids(tokenizer, block_size=16)
    scaffold_tokens = len(scaffold.header_token_ids) + len(scaffold.footer_token_ids)
    fact_tokens = len(select_memory_facts(tokenizer, _hits(1)).ordered_token_ids()[0])
    separator_tokens = len(separator_token_ids(tokenizer, SEPARATOR))
    with_both_separators = scaffold_tokens + fact_tokens + 2 * separator_tokens

    fitting = _serving_client(
        sample, inferscale=_kv_section_with_max_position(with_both_separators)
    )
    assert _answer(fitting, sample, _hits(1)).metrics["kv_memory_tokens"] == with_both_separators

    one_token_short = _serving_client(
        sample, inferscale=_kv_section_with_max_position(with_both_separators - 1)
    )
    with pytest.raises(RuntimeError) as error:
        _answer(one_token_short, sample, _hits(1))

    assert str(error.value) == (
        "Memory scaffold, context turns, and retrieved Mem0 facts cannot fit within the memory "
        f"token budget: required={with_both_separators} budget={with_both_separators - 1} "
        f"scaffold={scaffold_tokens + 2 * separator_tokens} context=0 facts={fact_tokens}."
    )


def test_blender_is_given_the_model_once_the_started_engine_has_loaded_it(
    lmcache: Runtime,
) -> None:
    client = _client()
    assert lmcache.events == []

    client.start_llm()
    client.start_llm()

    [engine] = lmcache.engines
    assert lmcache.events == ["engine_start", "register_model"]
    assert lmcache.registered_models == [engine.llm]


def test_load_statistics_cover_exactly_the_generation(lmcache: Runtime) -> None:
    sample = _sample()
    client = _serving_client(sample)
    lmcache.events.clear()

    _answer(client, sample, _hits())

    assert lmcache.events == ["reset", "generate", "snapshot"]


def test_latency_is_measured_from_the_start_times_the_caller_passes(lmcache: Runtime) -> None:
    sample = _sample()
    client = _serving_client(sample)
    retrieval_ms = 100
    ttft_started_at = lmcache.clock.perf_counter()
    lmcache.clock.advance_ms(30)

    metrics = _answer(
        client,
        sample,
        _hits(),
        ttft_started_at=ttft_started_at,
        query_started_at=ttft_started_at - retrieval_ms / 1000,
    ).metrics

    after_request_start_ms = 30 + _prompt_building_ms(sample, _hits()) + RESET_LOAD_STATS_MS
    assert metrics["answer_time_to_first_token_ms"] == pytest.approx(
        after_request_start_ms + ENGINE_TTFT_MS
    )
    assert metrics["query_to_first_token_ms"] == pytest.approx(
        retrieval_ms + after_request_start_ms + ENGINE_TTFT_MS
    )
    assert metrics["query_to_answer_ms"] == pytest.approx(
        retrieval_ms + after_request_start_ms + GENERATE_MS
    )


def test_warm_up_batches_256_prompts_because_each_prompt_is_a_single_segment(
    lmcache: Runtime,
) -> None:
    sample = _sample()
    client = _serving_client(sample)

    metrics = client.warm_sample_memory(sample, _hits())

    tokenizer = FakeTokenizer()
    separator = separator_token_ids(tokenizer, SEPARATOR)
    header = extract_memory_scaffold_token_ids(tokenizer, block_size=16).header_token_ids
    facts = select_memory_facts(tokenizer, _hits()).ordered_token_ids()
    [(prompts, batch_size)] = lmcache.warmed_batches
    assert batch_size == 256
    assert prompts == [header, *facts]
    assert not any(contains_separator(prompt, separator) for prompt in prompts)
    assert metrics == {"warmed": True}
    assert lmcache.events[-1] == "clear_cache"
