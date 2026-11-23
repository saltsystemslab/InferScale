from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from memory_config import make_memory_config

from benchmarks.common.clients import ChatResult
from benchmarks.common.vector_types import RetrievalMetrics, SearchHit
from benchmarks.memory import prediction
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, QuestionAnswer, Turn
from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.prediction import run_prediction_mode, scored_questions
from benchmarks.memory.runtime_clients import RuntimeClients

BUILD_RETRIEVER_MS = 50
PREPARE_SAMPLE_MS = 20
WARM_SAMPLE_MS = 300
RETRIEVAL_MS = 8
ANSWER_MS = 40
LOGGING_MS = 1
MEMORY_SETUP_MS = 4.0


class Clock:
    def __init__(self) -> None:
        self.now_s = 100.0

    def perf_counter(self) -> float:
        return self.now_s

    def advance_ms(self, milliseconds: float) -> None:
        self.now_s += milliseconds / 1000


class Pipeline:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, samples: list[ConversationSample]) -> None:
        self.clock = Clock()
        self.samples = samples
        self.events: list[tuple[Any, ...]] = []
        self.search_windows: dict[str, tuple[float, float]] = {}
        monkeypatch.setattr(prediction, "time", self.clock)
        monkeypatch.setattr(prediction, "load_locomo", self._load_locomo)
        monkeypatch.setattr(prediction, "SampleMemoryBuilder", self._builder)

    def _load_locomo(self, path: Path, max_samples: int | None = None) -> list[ConversationSample]:
        self.events.append(("load_locomo", path, max_samples))
        return self.samples if max_samples is None else self.samples[:max_samples]

    def _builder(self, config: MemoryRunConfig) -> Pipeline:
        return self

    def load_fact_catalog(self, sample: ConversationSample) -> tuple[MemoryFact, ...]:
        self.events.append(("load_fact_catalog", sample.sample_id))
        return (_fact(sample.sample_id),)

    def build_retriever_with_metrics(
        self, sample: ConversationSample
    ) -> tuple[Retriever, dict[str, Any]]:
        self.clock.advance_ms(BUILD_RETRIEVER_MS)
        self.events.append(("build_retriever", sample.sample_id))
        return Retriever(self, sample), {
            "memory_setup_time_ms": MEMORY_SETUP_MS,
            "memory_inferred_record_count": 1,
            "vector_backend": "qdrant",
        }

    def names(self) -> list[str]:
        return [event[0] for event in self.events]

    def of(self, name: str) -> list[tuple[Any, ...]]:
        return [event[1:] for event in self.events if event[0] == name]


class Retriever:
    def __init__(self, pipeline: Pipeline, sample: ConversationSample) -> None:
        self.pipeline = pipeline
        self.sample = sample

    def search(self, query: str, *, top_k: int) -> tuple[list[SearchHit], RetrievalMetrics]:
        clock = self.pipeline.clock
        started = clock.perf_counter()
        clock.advance_ms(RETRIEVAL_MS)
        self.pipeline.search_windows[query] = (started, clock.perf_counter())
        self.pipeline.events.append(("search", self.sample.sample_id, query, top_k))
        return [_fact(self.sample.sample_id).to_search_hit(1)], RetrievalMetrics(1.0, 2.0, 3.0)

    def log_cache_stats(self) -> None:
        self.pipeline.events.append(("log_cache_stats", self.sample.sample_id))

    def close(self) -> None:
        self.pipeline.events.append(("retriever_close", self.sample.sample_id))


class AnswerClient:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.request_clocks: dict[str, tuple[float | None, float | None]] = {}

    def start_llm(self) -> None:
        self.pipeline.events.append(("start_llm",))

    def prepare_sample(self, sample: ConversationSample) -> None:
        self.pipeline.clock.advance_ms(PREPARE_SAMPLE_MS)
        self.pipeline.events.append(("prepare_sample", sample.sample_id))

    def answer_with_retrieved_memory(
        self,
        *,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        ttft_started_at: float | None,
        query_started_at: float | None,
        **generation: Any,
    ) -> ChatResult:
        self.request_clocks[qa.question] = (query_started_at, ttft_started_at)
        self.pipeline.events.append(("answer", sample.sample_id, qa.question_id))
        self.pipeline.clock.advance_ms(ANSWER_MS)
        return ChatResult(content=f"ANSWER: reply to {qa.question}", ttft_ms=6.0)

    def close_sample(self) -> None:
        self.pipeline.events.append(("close_sample",))


class WarmingClient(AnswerClient):
    def warm_sample_memory(
        self, sample: ConversationSample, facts: list[SearchHit]
    ) -> dict[str, Any]:
        self.pipeline.clock.advance_ms(WARM_SAMPLE_MS)
        self.pipeline.events.append(("warm", sample.sample_id, [hit.id for hit in facts]))
        return {"lmcache_warm_time_ms": 9.0, "lmcache_warm_tokens": 42}


def _config(tmp_path: Path, **values: Any) -> MemoryRunConfig:
    settings: dict[str, Any] = {
        "dataset_path": tmp_path / "locomo.json",
        "results_dir": tmp_path / "results",
        "run_id": "predict",
        "answer_backend": "prompt-injection",
        "vector_backend": "qdrant",
        "skip_judge": True,
        "top_k": 3,
        "log_every": 0,
    }
    settings.update(values)
    return make_memory_config(**settings)


def _run(config: MemoryRunConfig, client: AnswerClient) -> prediction.PredictionResult:
    return run_prediction_mode(config, RuntimeClients(answer_client=client, judge_client=None))


def _sample(sample_id: str, categories: tuple[str, ...] = ("1", "2")) -> ConversationSample:
    turns = [
        Turn(
            sample_id=sample_id,
            session_id="session_1",
            session_index=1,
            turn_index=turn_index,
            speaker="Alice",
            text="I like tea.",
        )
        for turn_index in range(3)
    ]
    questions = [
        QuestionAnswer(
            sample_id=sample_id,
            question_id=f"q{index}",
            question=f"{sample_id} question {index}?",
            answer="tea",
            category=category,
        )
        for index, category in enumerate(categories)
    ]
    return ConversationSample(sample_id=sample_id, turns=turns, qa=questions, raw={})


def _fact(sample_id: str) -> MemoryFact:
    return MemoryFact(
        id=f"{sample_id}-fact",
        text="Alice likes tea.",
        created_at="2026-01-02T00:00:00+00:00",
        timestamp_epoch=1767312000,
        sample_id=sample_id,
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=0,
        source_turn_id=f"{sample_id}:session_1:0",
        speaker="Alice",
        role="user",
    )


def test_only_the_four_scored_categories_are_answered() -> None:
    sample = _sample("s1", categories=("1", "5", " 2 ", "adversarial", "4", "3"))
    numeric = QuestionAnswer("s1", "q9", "Numeric category?", "tea", 3)  # type: ignore[arg-type]

    assert [qa.question_id for qa in scored_questions(sample.qa)] == ["q0", "q2", "q4", "q5"]
    assert scored_questions([numeric]) == [numeric]
    assert scored_questions([]) == []


def test_prepare_time_covers_preparing_and_warming_the_sample_but_not_building_its_retriever(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = Pipeline(monkeypatch, [_sample("s1")])

    [row] = _run(_config(tmp_path), WarmingClient(pipeline)).sample_setup_metrics

    assert pipeline.names()[1:6] == [
        "load_fact_catalog",
        "start_llm",
        "build_retriever",
        "prepare_sample",
        "warm",
    ]
    assert pipeline.of("warm") == [("s1", ["s1-fact"])]
    assert row["answer_prepare_sample_time_ms"] == pytest.approx(PREPARE_SAMPLE_MS + WARM_SAMPLE_MS)
    assert row["lmcache_warm_time_ms"] == 9.0
    assert row["lmcache_warm_tokens"] == 42
    assert row["lmcache_warm_prompts"] is None


def test_query_clock_starts_before_retrieval_and_the_ttft_clock_right_after_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = Pipeline(monkeypatch, [_sample("s1")])
    client = AnswerClient(pipeline)
    slow_log = logger.add(lambda message: pipeline.clock.advance_ms(LOGGING_MS), level="INFO")

    try:
        _run(_config(tmp_path, log_every=1), client)
    finally:
        logger.remove(slow_log)

    assert client.request_clocks == pipeline.search_windows
    assert len(client.request_clocks) == 2
