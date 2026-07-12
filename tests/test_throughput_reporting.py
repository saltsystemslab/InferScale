from __future__ import annotations

from typing import Any

import pytest

from locomo_jasper_bench.throughput.config import parse_args
from locomo_jasper_bench.throughput.reporting import (
    RESULT_COLUMNS,
    merge_result_rows,
    render_markdown_report,
    validate_result_row,
)


def _row(condition: str, num_users: int, qps: float = 4.0, **overrides: Any) -> dict[str, Any]:
    total_requests = num_users * 2
    row: dict[str, Any] = {
        "run_id": "run",
        "model": "test/model",
        "model_label": "model",
        "condition": condition,
        "vector_backend": None if condition == "no_memory" else "jasper",
        "jasper_effective_beam_width": None if condition == "no_memory" else 64,
        "num_users": num_users,
        "memory_turn_count": 0.0 if condition == "no_memory" else 25.0,
        "requests_per_user": 2,
        "total_requests": total_requests,
        "throughput_qps": qps,
        "avg_latency_ms": 1000 / qps,
        "generation_time_s": total_requests / qps,
        "retrieval_time_s": 0.0,
        "vector_search_time_s": 0.0,
        "prompt_build_time_s": 0.0,
        "kv_compose_time_s": 0.0,
        "kv_verify_time_s": 0.0,
        "memory_setup_time_s": 0.0,
        "kv_precompute_time_s": 0.0,
        "engine_startup_time_s": 0.0,
        "kv_store_gpu_mb": 0.0,
        "kv_requests_loaded": 0 if condition != "kv_injection" else total_requests,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "input_tokens_per_second": 100.0,
        "output_tokens_per_second": 50.0,
    }
    row.update(overrides)
    return row


def test_result_columns_use_memory_turn_count() -> None:
    assert "memory_turn_count" in RESULT_COLUMNS
    assert "fact_count" not in RESULT_COLUMNS


def test_validate_result_row_requires_all_columns() -> None:
    row = _row("no_memory", 10)
    del row["memory_turn_count"]

    with pytest.raises(ValueError, match="memory_turn_count"):
        validate_result_row(row)


def test_merge_result_rows_replaces_by_condition_and_user_count() -> None:
    existing = [_row("no_memory", 10, qps=1.0), _row("mem0_jasper", 10, qps=2.0)]
    replacement = [_row("no_memory", 10, qps=9.0)]

    merged = merge_result_rows(existing, replacement)

    by_key = {(row["condition"], row["num_users"]): row for row in merged}
    assert by_key[("no_memory", 10)]["throughput_qps"] == 9.0
    assert by_key[("mem0_jasper", 10)]["throughput_qps"] == 2.0


def test_markdown_report_shows_turns_per_user() -> None:
    config, _ = parse_args(["--model", "test/model", "--user-counts", "10", "--run-id", "run"])
    rows = [
        _row("no_memory", 10),
        _row("mem0_jasper", 10),
        _row("kv_injection", 10, kv_verify_time_s=0.5),
    ]

    report = render_markdown_report(config, [validate_result_row(row) for row in rows])

    assert "Turns/user" in report
    assert "Facts/user" not in report
    assert "| 10 | 25 |" in report


def test_markdown_report_states_generation_only_qps() -> None:
    config, _ = parse_args(["--model", "test/model", "--user-counts", "10", "--run-id", "run"])
    rows = [_row("no_memory", 10)]

    report = render_markdown_report(config, [validate_result_row(row) for row in rows])

    assert "QPS for every condition times only the synchronous vLLM generation call" in report
    assert "UNUSED" not in report
