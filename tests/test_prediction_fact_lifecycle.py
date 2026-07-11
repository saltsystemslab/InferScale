from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from locomo_jasper_bench.clients import ChatResult
from locomo_jasper_bench.clients_factory import RuntimeClients
from locomo_jasper_bench.config import BenchmarkConfig
from locomo_jasper_bench.data import ConversationSample, QuestionAnswer, Turn
from locomo_jasper_bench.prediction import run_prediction_mode
from locomo_jasper_bench.retrieval.fact_catalog import MemoryFact
from locomo_jasper_bench.vector_types import RetrievalMetrics, SearchHit


def test_prediction_uses_fact_catalog_then_live_retriever_and_closes(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    events: list[tuple[str, Any]] = []
    sample = _sample()
    fact = _fact()
    retrieved_hit = fact.to_search_hit(1)

    class FakeRetriever:
        def __init__(self) -> None:
            self.memory = SimpleNamespace()

        def search(
            self,
            query: str,
            *,
            top_k: int,
        ) -> tuple[list[SearchHit], RetrievalMetrics]:
            events.append(("search", (query, top_k)))
            return [retrieved_hit], RetrievalMetrics(
                embedding_time_ms=1.0,
                search_time_ms=2.0,
                total_time_ms=3.0,
                vector_backend="jasper",
                jasper_effective_beam_width=64,
            )

        def close(self) -> None:
            events.append(("retriever_close", None))

    retriever = FakeRetriever()

    class FakeMemoryBuilder:
        def __init__(self, config: BenchmarkConfig) -> None:
            assert config.run_id == "fact-lifecycle"

        def load_fact_catalog(self, selected_sample: ConversationSample) -> tuple[MemoryFact, ...]:
            events.append(("load_catalog", selected_sample.sample_id))
            return (fact,)

        def build_retriever_with_metrics(
            self,
            selected_sample: ConversationSample,
        ) -> tuple[FakeRetriever, dict[str, Any]]:
            events.append(("build_retriever", selected_sample.sample_id))
            return retriever, {
                "memory_setup_time_ms": 4.0,
                "memory_inferred_record_count": 1,
            }

        def log_embedding_cache_stats(self, memory: Any, sample_id: str) -> None:
            assert memory is retriever.memory
            events.append(("log_cache", sample_id))

    class FakeAnswerClient:
        def precompute_sample_cache(
            self,
            selected_sample: ConversationSample,
            facts: list[SearchHit],
        ) -> dict[str, Any]:
            events.append(("kv_precompute", [hit.id for hit in facts]))
            assert selected_sample is sample
            assert facts[0].payload["source_turn_id"] == fact.source_turn_id
            return {"kv_precompute_time_ms": 5.0, "kv_precomputed_chunks": 1}

        def start_llm(self) -> None:
            events.append(("start_llm", None))

        def prepare_sample(self, selected_sample: ConversationSample) -> None:
            events.append(("prepare_sample", selected_sample.sample_id))

        def answer_with_retrieved_memory(
            self,
            *,
            sample: ConversationSample,
            qa: QuestionAnswer,
            hits: list[SearchHit],
            **_: Any,
        ) -> ChatResult:
            events.append(("answer", [hit.id for hit in hits]))
            assert sample.sample_id == "sample-1"
            assert qa.question_id == "q1"
            assert hits == [retrieved_hit]
            return ChatResult(content="ANSWER: tea", ttft_ms=6.0)

        def close_sample(self) -> None:
            events.append(("close_sample", None))

    monkeypatch.setattr(
        "locomo_jasper_bench.prediction.load_locomo",
        lambda *_args, **_kwargs: [sample],
    )
    monkeypatch.setattr(
        "locomo_jasper_bench.prediction.SampleMemoryBuilder",
        FakeMemoryBuilder,
    )
    config = BenchmarkConfig(
        dataset_path=tmp_path / "unused.json",
        results_dir=tmp_path / "results",
        run_id="fact-lifecycle",
        answer_backend="vllm-kv",
        vector_backend="jasper",
        top_k=1,
        max_samples=1,
        skip_judge=True,
        log_every=0,
    )

    result = run_prediction_mode(
        config,
        RuntimeClients(answer_client=FakeAnswerClient(), judge_client=None),
    )

    event_names = [name for name, _ in events]
    assert event_names.index("load_catalog") < event_names.index("kv_precompute")
    assert event_names.index("kv_precompute") < event_names.index("start_llm")
    assert event_names.index("start_llm") < event_names.index("build_retriever")
    assert event_names.index("build_retriever") < event_names.index("search")
    assert event_names.index("search") < event_names.index("answer")
    assert event_names[-3:] == ["log_cache", "retriever_close", "close_sample"]
    assert result.records[0]["predicted_answer"] == "tea"
    assert result.records[0]["retrieved_memories"][0]["id"] == fact.id
    assert result.records[0]["retrieved_memories"][0]["source_turn_id"] == fact.source_turn_id
    assert result.records[0]["metrics"]["query_retrieval_time_ms"] == 3.0


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
        qa=[
            QuestionAnswer(
                sample_id="sample-1",
                question_id="q1",
                question="What does Alice like?",
                answer="tea",
                category="1",
            )
        ],
        raw={"conversation": {"speaker_a": "Alice", "speaker_b": "Bob"}},
    )


def _fact() -> MemoryFact:
    return MemoryFact(
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
