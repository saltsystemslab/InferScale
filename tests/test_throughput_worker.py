from __future__ import annotations

import pytest

from locomo_jasper_bench.throughput.config import parse_args
from locomo_jasper_bench.throughput.reporting import RESULT_COLUMNS, validate_result_row
from locomo_jasper_bench.throughput.worker import (
    _request_question_answer,
    _require_canonical_memory_tokens,
    _result_row,
)
from locomo_jasper_bench.throughput.workload import LocomoRequest


def _config():
    config, _ = parse_args(
        ["--model", "test/model", "--user-counts", "10", "--run-id", "run"]
    )
    return config


def test_result_row_matches_the_report_schema() -> None:
    row = _result_row(
        _config(),
        10,
        condition="mem0_jasper",
        vector_backend="jasper",
        jasper_effective_beam_width=64,
        memory_turn_count=25.0,
        generation_time_s=4.0,
        retrieval_time_s=0.5,
        prompt_build_time_s=0.5,
        total_input_tokens=1000,
        total_output_tokens=500,
    )

    assert tuple(row) == RESULT_COLUMNS
    assert row["total_requests"] == 20
    assert row["throughput_qps"] == pytest.approx(5.0)
    assert row["avg_latency_ms"] == pytest.approx(200.0)
    assert row["memory_turn_count"] == 25.0
    assert row["kv_requests_loaded"] == 0
    assert validate_result_row(row) is not None


def test_result_row_qps_counts_only_generation_time() -> None:
    row = _result_row(
        _config(),
        10,
        condition="kv_injection",
        vector_backend="jasper",
        generation_time_s=2.0,
        retrieval_time_s=50.0,
        kv_compose_time_s=10.0,
        prompt_build_time_s=5.0,
        kv_requests_loaded=20,
        total_input_tokens=1000,
        total_output_tokens=500,
    )

    assert row["throughput_qps"] == pytest.approx(10.0)
    assert row["kv_requests_loaded"] == 20


def test_result_row_rejects_non_positive_times() -> None:
    with pytest.raises(RuntimeError, match="greater than zero"):
        _result_row(
            _config(),
            10,
            condition="no_memory",
            generation_time_s=0.0,
            total_input_tokens=1,
            total_output_tokens=1,
        )


def test_request_question_answer_carries_request_identity() -> None:
    request = LocomoRequest(
        user_id="user_0003",
        user_index=3,
        sample_id="sample-1",
        question_id="q-7",
        query="What does Alice like?",
    )

    qa = _request_question_answer(request)

    assert qa.sample_id == "sample-1"
    assert qa.question_id == "q-7"
    assert qa.question == "What does Alice like?"


def test_canonical_memory_token_mismatch_raises_with_index() -> None:
    _require_canonical_memory_tokens([1, 2, 3], [1, 2, 3])

    with pytest.raises(RuntimeError, match="index=1"):
        _require_canonical_memory_tokens([1, 9, 3], [1, 2, 3])
