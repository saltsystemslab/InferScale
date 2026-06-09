from __future__ import annotations

import pytest

from locomo_jasper_bench.clients import ChatResult
from locomo_jasper_bench.config import BenchmarkConfig
from locomo_jasper_bench.data import ConversationSample, QuestionAnswer
from locomo_jasper_bench.results import summarize_records
from locomo_jasper_bench.runner import QuestionEvaluator, RuntimeClients, run_benchmark
from locomo_jasper_bench.vector_types import SearchHit


def test_summarize_records_includes_exact_top_k_answer_accuracy() -> None:
    rows = [
        _row(jasper_correct=True, exact_correct=False),
        _row(jasper_correct=False, exact_correct=True),
        _row(jasper_correct=False, exact_correct=True),
    ]

    summary = summarize_records(rows, run_id="run", mode="baseline", config={}, system_metadata={})

    metrics = summary["metrics"]
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["exact_top_k_answer_accuracy"] == pytest.approx(2 / 3)
    assert metrics["exact_top_k_judged_count"] == 3
    assert metrics["exact_top_k_correct_count"] == 2
    assert metrics["exact_top_k_paired_judged_count"] == 3
    assert metrics["answer_accuracy_delta_exact_minus_jasper"] == pytest.approx(1 / 3)
    assert metrics["exact_correct_jasper_wrong_count"] == 2
    assert metrics["jasper_correct_exact_wrong_count"] == 1


def test_exact_answer_baseline_requires_jasper_backend(tmp_path) -> None:
    config = BenchmarkConfig(
        results_dir=tmp_path,
        vector_backend="qdrant",
        exact_answer_baseline=True,
    )

    with pytest.raises(ValueError, match="--exact-answer-baseline"):
        run_benchmark(config, clients=None)


def test_exact_top_k_answer_records_answer_judge_retrieval_and_metrics() -> None:
    answer_client = _SequencedChatClient([ChatResult("exact answer", ttft_ms=12.5)])
    judge_client = _SequencedChatClient([ChatResult('{"correct": true, "reason": "matches"}')])
    evaluator = QuestionEvaluator(
        BenchmarkConfig(top_k=2, exact_answer_baseline=True),
        RuntimeClients(answer_client=answer_client, judge_client=judge_client),
    )
    vector_store = _ExactVectorStore()
    memory = _Memory(vector_store)
    sample = ConversationSample(sample_id="sample-1", turns=[], qa=[], raw={})
    qa = QuestionAnswer(
        sample_id="sample-1",
        question_id="q1",
        question="Who was mentioned?",
        answer="Alice",
        category="test",
    )

    result = evaluator._exact_top_k_answer(
        sample=sample,
        qa=qa,
        memory=memory,
        query_embedding=[0.1, 0.2],
    )

    assert vector_store.requested_top_k == 2
    assert result["predicted_answer"] == "exact answer"
    assert result["judge"] == {"correct": True, "reason": "matches", "raw": '{"correct": true, "reason": "matches"}'}
    assert result["retrieved_memories"] == [
        {
            "id": "hit-1",
            "rank": 1,
            "score": 0.9,
            "distance": 0.1,
            "memory": "Alice was mentioned.",
            "metadata": {"dia_id": "D1:1"},
        }
    ]
    assert result["metrics"]["answer_time_to_first_token_ms"] == 12.5
    assert result["metrics"]["exact_vector_db_query_time_ms"] >= 0.0


def _row(*, jasper_correct: bool, exact_correct: bool) -> dict[str, object]:
    return {
        "judge": {"correct": jasper_correct},
        "metrics": {"vector_db_query_time_ms": 1.0},
        "exact_top_k_answer": {
            "judge": {"correct": exact_correct},
            "metrics": {"exact_vector_db_query_time_ms": 2.0},
        },
    }


class _SequencedChatClient:
    def __init__(self, results: list[ChatResult]) -> None:
        self._results = results
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        ttft_started_at: float | None = None,
    ) -> ChatResult:
        self.calls.append(messages)
        return self._results.pop(0)


class _Memory:
    def __init__(self, vector_store: object) -> None:
        self.vector_store = vector_store


class _ExactVectorStore:
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    def exact_search(
        self,
        *,
        query: str,
        vectors: list[float],
        top_k: int,
    ) -> list[SearchHit]:
        self.requested_top_k = top_k
        return [
            SearchHit(
                id="hit-1",
                payload={"memory": "Alice was mentioned.", "metadata": {"dia_id": "D1:1"}},
                score=0.9,
                distance=0.1,
                rank=1,
            )
        ]
