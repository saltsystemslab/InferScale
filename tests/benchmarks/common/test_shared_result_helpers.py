from __future__ import annotations

from benchmarks.common.answer_timing import retrieval_latency_metrics
from benchmarks.common.cache_keys import payload_header_matches
from benchmarks.common.clients import ChatResult
from benchmarks.common.judge import judge_with
from benchmarks.common.run_summary import summary_envelope
from inferscale.v1.types import RetrievalMetrics


class RecordingJudge:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[list[dict[str, str]], int, float, float]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        del response_format
        self.calls.append((messages, max_tokens, temperature, top_p))
        return ChatResult(content=self.reply)


def test_judge_with_is_deterministic_and_carries_metadata() -> None:
    judge = RecordingJudge("true")
    messages = [{"role": "user", "content": "Is it so?"}]

    payload = judge_with(judge, messages, max_tokens=4, metadata={"model": "judge-model"})

    assert payload["correct"] is True
    assert payload["status"] == "ok"
    assert payload["model"] == "judge-model"
    assert judge.calls == [(messages, 4, 0.0, 1.0)]


def test_retrieval_latency_metrics_report_the_backend_only_when_known() -> None:
    timings = RetrievalMetrics(embedding_time_ms=1.0, search_time_ms=2.0, total_time_ms=3.5)
    assert retrieval_latency_metrics(timings) == {
        "query_embedding_time_ms": 1.0,
        "vector_db_query_time_ms": 2.0,
        "query_retrieval_time_ms": 3.5,
    }

    detailed = RetrievalMetrics(
        embedding_time_ms=1.0,
        search_time_ms=2.0,
        total_time_ms=3.5,
        vector_backend="jasper",
        jasper_effective_beam_width=64,
    )
    metrics = retrieval_latency_metrics(detailed)
    assert metrics["resolved_vector_backend"] == "jasper"
    assert metrics["jasper_effective_beam_width"] == 64


def test_summary_envelope_counts_judged_records_and_orders_sections() -> None:
    records = [
        {"judge": {"correct": True}},
        {"judge": {"correct": False}},
        {"judge": {"correct": None}},
        {},
    ]

    summary = summary_envelope(
        records,
        run_id="run",
        mode="mode",
        metrics={"accuracy": 0.5},
        config={"setting": 1},
        system_metadata={"gpu": None},
        setup={"chunk_count": 2},
    )

    assert list(summary) == [
        "run_id",
        "mode",
        "question_count",
        "judged_count",
        "correct_count",
        "metrics",
        "setup",
        "config",
        "system",
    ]
    assert (summary["question_count"], summary["judged_count"], summary["correct_count"]) == (
        4,
        2,
        1,
    )


def test_payload_header_matches_checks_version_and_every_identity_value() -> None:
    payload = {"version": 2, "meta": {"model": "m", "dtype": "bf16"}}

    assert payload_header_matches(payload, version=2, expected_meta={"model": "m"})
    assert not payload_header_matches(payload, version=1, expected_meta={"model": "m"})
    assert not payload_header_matches(payload, version=2, expected_meta={"model": "other"})
    assert not payload_header_matches({"version": 2, "meta": None}, version=2, expected_meta={})
    assert not payload_header_matches("not a payload", version=2, expected_meta={})
