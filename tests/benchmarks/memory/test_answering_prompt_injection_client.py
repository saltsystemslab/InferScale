from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from library_fakes import FakeTokenizer
from memory_config import make_memory_config

from benchmarks.memory import answer_prompt_injection
from benchmarks.memory.answer_prompt_injection import PromptInjectionAnswerClient
from benchmarks.memory.data import ConversationSample, QuestionAnswer, Turn
from benchmarks.memory.mem0.fact_catalog import MemoryFact, fact_catalog_hits
from benchmarks.memory.prompting import build_memory_prompt_token_ids, build_memory_query
from inferscale.v1.serving.vllm import GenerationOutput
from inferscale.v1.types import SearchHit

TOKENIZE_MS = 1
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
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.tokenizer = TimedTokenizer(clock)
        self.started = False
        self.prompts: list[list[int]] = []
        self.sampling: list[tuple[int, float, float]] = []

    def start(self, *, kv_transfer_config: Any = None) -> float:
        self.started = True
        return 0.0

    def generate(
        self, prompt_token_ids: Any, *, max_tokens: int, temperature: float, top_p: float
    ) -> GenerationOutput:
        self.prompts.append(list(prompt_token_ids))
        self.sampling.append((max_tokens, temperature, top_p))
        started_at = self.clock.perf_counter()
        self.clock.advance_ms(GENERATE_MS)
        return GenerationOutput(
            text="tea",
            ttft_ms=ENGINE_TTFT_MS,
            started_at=started_at,
            finished_at=self.clock.perf_counter(),
        )


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> Engine:
    clock = Clock()
    engine = Engine(clock)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setattr(answer_prompt_injection, "time", clock)
    monkeypatch.setattr(answer_prompt_injection, "VLLMEngine", lambda *_: engine)
    return engine


def _client(**values: Any) -> PromptInjectionAnswerClient:
    config = make_memory_config(
        answer_backend="prompt-injection", vector_backend="qdrant", skip_judge=True, **values
    )
    return PromptInjectionAnswerClient(config)


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
    question = QuestionAnswer(
        sample_id=sample_id,
        question_id="q0",
        question="What does Alice like?",
        answer="tea",
        category="1",
    )
    return ConversationSample(sample_id=sample_id, turns=turns, qa=[question], raw={})


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


def _hits() -> list[SearchHit]:
    return fact_catalog_hits([_fact(0), _fact(1)])


def _tokenizing_ms(tokenize: Callable[[TimedTokenizer], Any]) -> float:
    clock = Clock()
    started = clock.perf_counter()
    tokenize(TimedTokenizer(clock))
    return (clock.perf_counter() - started) * 1000


def _prompt_building_ms(sample: ConversationSample) -> float:
    def build_prompt(tokenizer: TimedTokenizer) -> None:
        query = build_memory_query(
            tokenizer,
            sample.qa[0],
            block_size=16,
            max_position=32768,
            max_model_len=32768,
            max_answer_tokens=16,
        )
        build_memory_prompt_token_ids(
            tokenizer, sample, _hits(), memory_scaffold=query.scaffold, render_context_turns=True
        )

    return _tokenizing_ms(build_prompt)


def _answer(client: PromptInjectionAnswerClient, sample: ConversationSample, **options: Any):
    return client.answer_with_retrieved_memory(
        sample=sample,
        qa=sample.qa[0],
        hits=_hits(),
        **{"max_tokens": 16, "temperature": 0.0, "top_p": 1.0, **options},
    )


@pytest.mark.parametrize("context_window", [0, 1])
def test_prompt_renders_each_fact_after_its_context_turns(
    engine: Engine, context_window: int
) -> None:
    sample = _sample()
    client = _client(context_window=context_window)
    client.start_llm()
    client.prepare_sample(sample)

    result = _answer(client, sample, max_tokens=64, temperature=0.5, top_p=0.9)

    tokenizer = FakeTokenizer()
    query = build_memory_query(
        tokenizer,
        sample.qa[0],
        block_size=16,
        max_position=32768,
        max_model_len=32768,
        max_answer_tokens=64,
    )
    memory = build_memory_prompt_token_ids(
        tokenizer,
        sample,
        _hits(),
        context_window=context_window,
        memory_token_budget=query.memory_token_budget,
        memory_scaffold=query.scaffold,
        render_context_turns=True,
    )
    assert engine.prompts == [[*memory.token_ids, *query.query_tokens.token_ids]]
    assert engine.sampling == [(64, 0.5, 0.9)]
    assert result.content == "tea"
    assert result.ttft_ms == ENGINE_TTFT_MS
    metrics = result.metrics
    assert metrics["kv_memory_tokens"] == len(memory.token_ids)
    assert metrics["kv_query_tokens"] == len(query.query_tokens.token_ids)
    assert metrics["kv_query_bos_stripped"] == 0
    assert metrics["kv_selected_fact_ids"] == ["fact-1", "fact-0"]
    assert metrics["kv_block_size"] == 16
    assert metrics["kv_context_window"] == context_window
    assert metrics["memory_context_turn_count"] == context_window
    assert metrics["memory_context_text_tokens"] == memory.fact_plan.context_text_tokens
    assert metrics["memory_token_budget"] == query.memory_token_budget
    assert sorted(metrics) == [
        "answer_generate_time_ms",
        "answer_time_to_first_token_ms",
        "answer_total_time_ms",
        "kv_block_size",
        "kv_context_window",
        "kv_memory_tokens",
        "kv_query_bos_stripped",
        "kv_query_tokens",
        "kv_selected_fact_ids",
        "memory_context_encoding_tokens_max",
        "memory_context_encoding_tokens_total",
        "memory_context_encoding_truncated_tokens",
        "memory_context_text_tokens",
        "memory_context_turn_count",
        "memory_context_turn_ids",
        "memory_context_window",
        "memory_injected_fact_ids",
        "memory_retrieved_fact_ids",
        "memory_retrieved_fact_text_hashes",
        "memory_token_budget",
        "prompt_injection_engine_time_to_first_token_ms",
    ]


def test_latency_is_measured_from_the_start_times_the_caller_passes(engine: Engine) -> None:
    sample = _sample()
    client = _client()
    client.start_llm()
    client.prepare_sample(sample)
    retrieval_ms = 100
    ttft_started_at = engine.clock.perf_counter()
    engine.clock.advance_ms(30)

    metrics = _answer(
        client,
        sample,
        ttft_started_at=ttft_started_at,
        query_started_at=ttft_started_at - retrieval_ms / 1000,
    ).metrics

    after_request_start_ms = 30 + _prompt_building_ms(sample)
    assert metrics["answer_time_to_first_token_ms"] == pytest.approx(
        after_request_start_ms + ENGINE_TTFT_MS
    )
    assert metrics["query_to_first_token_ms"] == pytest.approx(
        retrieval_ms + after_request_start_ms + ENGINE_TTFT_MS
    )
    assert metrics["query_to_answer_ms"] == pytest.approx(
        retrieval_ms + after_request_start_ms + GENERATE_MS
    )
