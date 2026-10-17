from __future__ import annotations

import pytest

from benchmarks.common.answer_timing import generation_latency_metrics
from inferscale.v1.serving.vllm import GenerationOutput


def _output() -> GenerationOutput:
    return GenerationOutput(text="answer", ttft_ms=50.0, started_at=10.0, finished_at=10.3)


def test_latency_metrics_measure_from_request_and_query_starts() -> None:
    metrics = generation_latency_metrics(
        _output(),
        request_started=9.9,
        query_started_at=9.8,
        engine_ttft_key="kv_engine_time_to_first_token_ms",
    )

    assert metrics["answer_generate_time_ms"] == pytest.approx(300.0)
    assert metrics["answer_total_time_ms"] == pytest.approx(400.0)
    assert metrics["kv_engine_time_to_first_token_ms"] == 50.0
    assert metrics["answer_time_to_first_token_ms"] == pytest.approx(150.0)
    assert metrics["query_to_answer_ms"] == pytest.approx(500.0)
    assert metrics["query_to_first_token_ms"] == pytest.approx(250.0)


def test_excluded_bookkeeping_is_not_charged_to_latency() -> None:
    metrics = generation_latency_metrics(
        _output(),
        request_started=9.9,
        query_started_at=None,
        engine_ttft_key="prompt_injection_engine_time_to_first_token_ms",
        excluded_ms=60.0,
    )

    assert metrics["answer_total_time_ms"] == pytest.approx(340.0)
    assert metrics["answer_time_to_first_token_ms"] == pytest.approx(90.0)
    assert metrics["prompt_injection_engine_time_to_first_token_ms"] == 50.0
    assert "query_to_answer_ms" not in metrics
    assert "query_to_first_token_ms" not in metrics


def test_latency_never_goes_negative_when_exclusions_exceed_the_window() -> None:
    metrics = generation_latency_metrics(
        _output(),
        request_started=9.95,
        query_started_at=None,
        engine_ttft_key="ttft",
        excluded_ms=1000.0,
    )

    assert metrics["answer_total_time_ms"] == 0.0
    assert metrics["answer_time_to_first_token_ms"] == 50.0
