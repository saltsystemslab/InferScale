from __future__ import annotations

import csv
from pathlib import Path

from locomo_jasper_bench.reporting import QUERY_METRICS_COLUMNS, query_metric_rows, write_csv
from locomo_jasper_bench.results import summarize_records


def _record(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "run_id": "run",
        "mode": "full",
        "sample_id": "sample",
        "question_id": "question",
        "category": "single-hop",
        "retrieved_memories": [{"turn_id": "turn-2"}, {"turn_id": "turn-4"}],
        "metrics": metrics,
        "judge": {"correct": True},
    }


def _judged_record(category: str, correct: bool | None) -> dict[str, object]:
    record = _record({})
    record["category"] = category
    record["judge"] = {"correct": correct}
    return record


def test_summary_reports_accuracy_by_upstream_category() -> None:
    rows = [
        _judged_record("1", True),
        _judged_record("1", False),
        _judged_record("2", True),
        _judged_record("4", None),
    ]

    summary = summarize_records(
        rows,
        run_id="run",
        mode="full",
        config={},
        system_metadata={},
    )
    by_category = summary["metrics"]["accuracy_by_category"]

    assert by_category["1"] == {
        "name": "multi-hop",
        "total": 2,
        "correct": 1,
        "accuracy": 0.5,
    }
    assert by_category["2"] == {
        "name": "temporal",
        "total": 1,
        "correct": 1,
        "accuracy": 1.0,
    }
    assert "4" not in by_category  # unjudged records are excluded


def test_summary_aggregates_kv_verify_time() -> None:
    rows = [
        _record({"kv_verify_time_ms": 10.0}),
        _record({"kv_verify_time_ms": 30.0}),
    ]

    summary = summarize_records(
        rows,
        run_id="run",
        mode="full",
        config={},
        system_metadata={},
    )

    assert summary["metrics"]["kv_verify_time_ms"]["avg"] == 20.0


def test_query_metrics_expose_backend_neutral_memory_audit_fields(tmp_path: Path) -> None:
    record = _record(
        {
            "kv_memory_tokens": 40,
            "kv_query_tokens": 10,
            "memory_context_window": 2,
            "memory_retrieved_fact_ids": ["fact-2", "fact-4"],
            "memory_retrieved_fact_text_hashes": ["sha256:two", "sha256:four"],
            "memory_injected_fact_ids": ["fact-4", "fact-2"],
            "memory_context_turn_ids": ["turn-1", "turn-3"],
            "memory_context_turn_count": 2,
            "memory_context_encoding_tokens_total": 81,
            "memory_context_encoding_tokens_max": 43,
            "memory_context_encoding_truncated_tokens": 0,
            "memory_token_budget": 64,
            "kv_block_size": 16,
            "kv_loaded_memory_tokens": 32,
            "kv_recomputed_memory_tail_tokens": 8,
            "kv_fact_tokens_end": 30,
            "kv_verify_time_ms": 12.5,
        }
    )

    row = query_metric_rows([record])[0]

    assert row["memory_tokens"] == 40
    assert row["query_tokens"] == 10
    assert row["total_prompt_tokens"] == 50
    assert row["memory_context_window"] == 2
    assert row["memory_retrieved_fact_ids"] == ["fact-2", "fact-4"]
    assert row["memory_retrieved_fact_text_hashes"] == ["sha256:two", "sha256:four"]
    assert row["memory_injected_fact_ids"] == ["fact-4", "fact-2"]
    assert row["memory_context_turn_ids"] == ["turn-1", "turn-3"]
    assert row["memory_context_turn_count"] == 2
    assert row["memory_context_encoding_tokens_total"] == 81
    assert row["memory_context_encoding_tokens_max"] == 43
    assert row["memory_context_encoding_truncated_tokens"] == 0
    assert row["memory_token_budget"] == 64
    assert row["kv_block_size"] == 16
    assert row["kv_loaded_memory_tokens"] == 32
    assert row["kv_recomputed_memory_tail_tokens"] == 8
    assert row["kv_fact_tokens_end"] == 30
    assert row["kv_verify_time_ms"] == 12.5

    path = tmp_path / "query_metrics.csv"
    write_csv(path, [row], QUERY_METRICS_COLUMNS)
    with path.open(newline="", encoding="utf-8") as fh:
        written = next(csv.DictReader(fh))
    assert written["memory_injected_fact_ids"] == '["fact-4","fact-2"]'


def test_historical_kv_metrics_remain_readable() -> None:
    record = _record(
        {
            "kv_memory_tokens": 31,
            "kv_query_tokens": 11,
            "kv_context_window": 3,
        }
    )

    row = query_metric_rows([record])[0]
    summary = summarize_records(
        [record],
        run_id="run",
        mode="full",
        config={},
        system_metadata={},
    )

    assert row["memory_tokens"] == 31
    assert row["query_tokens"] == 11
    assert row["memory_context_window"] == 3
    assert summary["metrics"]["kv_context_window"]["avg"] == 3


def test_summary_aggregates_backend_neutral_memory_audit_metrics() -> None:
    rows = [
        _record(
            {
                "memory_context_window": 2,
                "memory_context_turn_count": 2,
                "memory_context_encoding_tokens_total": 41,
                "memory_context_encoding_tokens_max": 23,
                "memory_context_encoding_truncated_tokens": 0,
                "memory_token_budget": 64,
            }
        ),
        _record(
            {
                "memory_context_window": 2,
                "memory_context_turn_count": 2,
                "memory_context_encoding_tokens_total": 55,
                "memory_context_encoding_tokens_max": 29,
                "memory_context_encoding_truncated_tokens": 4,
                "memory_token_budget": 64,
            }
        ),
    ]

    summary = summarize_records(
        rows,
        run_id="run",
        mode="full",
        config={},
        system_metadata={},
    )
    metrics = summary["metrics"]

    assert metrics["memory_context_window"]["avg"] == 2
    assert metrics["memory_context_turn_count"]["avg"] == 2
    assert metrics["memory_context_encoding_tokens_total"]["avg"] == 48
    assert metrics["memory_context_encoding_tokens_max"]["max"] == 29
    assert metrics["memory_context_encoding_truncated_tokens"]["avg"] == 2
    assert metrics["memory_token_budget"]["avg"] == 64
