from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.reporting import (
    RESULT_COLUMNS,
    _coerce_row,
    build_result_row,
    merge_result_rows,
    read_existing_results,
    write_reports,
)


def _config(tmp_path: Path) -> ThroughputConfig:
    return make_throughput_config(
        tmp_path,
        run_id="report-test",
        conditions=("mem0_jasper", "kv_injection"),
        user_counts=(10,),
    )


def _row(condition: str, qps: float, *, num_users: int = 10) -> dict[str, object]:
    is_kv = condition == "kv_injection"
    uses_jasper = condition in {"mem0_jasper", "kv_injection"}
    values: dict[str, object] = {
        "run_id": "report-test",
        "model": "test/model",
        "model_label": "test",
        "condition": condition,
        "vector_backend": "jasper" if uses_jasper else None,
        "jasper_effective_beam_width": 64 if uses_jasper else None,
        "num_users": num_users,
        "fact_count": 200.0 if uses_jasper else 0.0,
        "requests_per_user": 2,
        "total_requests": num_users * 2,
        "throughput_qps": qps,
        "avg_latency_ms": 1000 / qps,
        "generation_time_s": num_users * 2 / qps,
        "retrieval_time_s": 0.2 if uses_jasper else 0.0,
        "vector_search_time_s": 0.05 if uses_jasper else 0.0,
        "prompt_build_time_s": 0.1,
        "kv_compose_time_s": 0.3 if is_kv else 0.0,
        "kv_verify_time_s": 0.05 if is_kv else 0.0,
        "memory_setup_time_s": 0.5 if uses_jasper else 0.0,
        "kv_precompute_time_s": 1.0 if is_kv else 0.0,
        "engine_startup_time_s": 2.0,
        "kv_prefix_caching": 1,
        "kv_store_gpu_mb": 10.0 if is_kv else 0.0,
        "kv_store_backend": "cpu" if is_kv else "gpu",
        "kv_store_host_mb": 120.0 if is_kv else 0.0,
        "kv_store_write_time_s": 0.4 if is_kv else 0.0,
        "kv_h2d_bytes": 125_829_120 if is_kv else 0,
        "kv_h2d_avg_ms": 5.5 if is_kv else 0.0,
        "kv_h2d_p95_ms": 9.0 if is_kv else 0.0,
        "kv_h2d_overlap_ratio": 0.8 if is_kv else 0.0,
        "kv_staging_stall_ms": 12.0 if is_kv else 0.0,
        "kv_requests_loaded": num_users * 2 if is_kv else 0,
        "total_input_tokens": 10240,
        "total_output_tokens": 1000,
        "input_tokens_per_second": 5120.0,
        "output_tokens_per_second": 500.0,
    }
    assert tuple(values) == RESULT_COLUMNS
    return values


def test_merge_result_rows_replaces_same_condition_and_user_count() -> None:
    merged = merge_result_rows([_row("kv_injection", 10.0)], [_row("kv_injection", 12.0)])

    assert len(merged) == 1
    assert merged[0]["throughput_qps"] == 12.0


def test_merge_keeps_distinct_user_counts_and_conditions_separate() -> None:
    merged = merge_result_rows(
        [_row("mem0_jasper", 10.0), _row("kv_injection", 12.0)],
        [_row("kv_injection", 14.0, num_users=25)],
    )

    assert {(row["condition"], row["num_users"]) for row in merged} == {
        ("mem0_jasper", 10),
        ("kv_injection", 10),
        ("kv_injection", 25),
    }


def test_write_reports_creates_csv_json_and_markdown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = write_reports(
        config,
        [_row("mem0_jasper", 10.0), _row("kv_injection", 20.0)],
        system_metadata={"gpu": {"available": False}},
    )

    assert summary["row_count"] == 2
    assert summary["user_counts"] == [10]
    with (config.run_dir / "throughput_merged.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all("fact_count" in row and "kv_verify_time_s" in row for row in rows)
    saved_summary = json.loads((config.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["conditions"]["kv_injection"]["maximum_qps"] == 20.0
    report = (config.run_dir / "throughput_report.md").read_text(encoding="utf-8")
    assert "KV / Mem0 Jasper" in report
    assert "Facts/user" in report
    assert "2.00x" in report
    assert "KV verify (s)" in report
    assert "QPS for every condition times only the synchronous vLLM generation call" in report


def test_coerce_row_parses_csv_strings_into_schema_types() -> None:
    def as_csv(row: dict[str, object]) -> dict[str, str]:
        return {column: "" if value is None else str(value) for column, value in row.items()}

    kv = _coerce_row(as_csv(_row("kv_injection", 10.0)))
    assert kv["num_users"] == 10 and isinstance(kv["num_users"], int)
    assert kv["throughput_qps"] == 10.0 and isinstance(kv["throughput_qps"], float)
    assert kv["jasper_effective_beam_width"] == 64
    assert kv["kv_store_backend"] == "cpu"

    qdrant = _coerce_row(as_csv(_row("mem0_qdrant", 5.0)))
    assert qdrant["vector_backend"] is None
    assert qdrant["jasper_effective_beam_width"] is None


def test_coerce_row_rejects_missing_columns() -> None:
    row = _row("kv_injection", 10.0)
    row.pop("kv_h2d_bytes")

    with pytest.raises(ValueError, match="missing columns"):
        _coerce_row(row)


def test_result_rows_round_trip_through_condition_csvs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    qdrant = build_result_row(
        config,
        10,
        condition="mem0_qdrant",
        vector_backend="qdrant",
        generation_time_s=2.0,
        retrieval_time_s=0.5,
        vector_search_time_s=0.3,
        total_input_tokens=100,
        total_output_tokens=40,
    )
    write_reports(config, [qdrant, _row("mem0_jasper", 12.0)], system_metadata={})

    restored, jasper = read_existing_results(config.run_dir)
    assert restored == qdrant
    assert restored["throughput_qps"] == 10.0
    assert restored["avg_latency_ms"] == 100.0
    assert restored["retrieval_time_s"] == 0.5
    assert restored["vector_search_time_s"] == 0.3
    assert jasper["throughput_qps"] == 12.0
    assert jasper["jasper_effective_beam_width"] == 64
