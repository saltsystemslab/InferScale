from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.reporting import (
    RESULT_COLUMNS,
    _coerce_row,
    build_result_row,
    condition_csv_path,
    merge_result_rows,
    read_existing_results,
    write_reports,
)


def _config(tmp_path: Path) -> ThroughputConfig:
    return ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
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
        "qdrant_deepcopy_time_ms": None,
        "qdrant_deepcopy_calls": None,
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
        "qdrant_query_deepcopy_time_ms": None,
        "qdrant_query_deepcopy_calls": None,
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


def test_coerce_row_tolerates_pre_change_csv_rows() -> None:
    legacy = _row("kv_injection", 10.0)
    for column in (
        "kv_prefix_caching",
        "kv_store_backend",
        "kv_store_host_mb",
        "kv_store_write_time_s",
        "kv_h2d_bytes",
        "kv_h2d_avg_ms",
        "kv_h2d_p95_ms",
        "kv_h2d_overlap_ratio",
        "kv_staging_stall_ms",
        "qdrant_deepcopy_time_ms",
        "qdrant_deepcopy_calls",
        "qdrant_query_deepcopy_time_ms",
        "qdrant_query_deepcopy_calls",
    ):
        legacy.pop(column)

    coerced = _coerce_row(legacy)
    assert coerced["kv_h2d_bytes"] == 0
    assert coerced["kv_prefix_caching"] == 0
    assert coerced["kv_store_backend"] == "gpu"
    assert coerced["qdrant_deepcopy_time_ms"] is None
    assert coerced["qdrant_deepcopy_calls"] is None
    assert coerced["qdrant_query_deepcopy_time_ms"] is None
    assert coerced["qdrant_query_deepcopy_calls"] is None


@pytest.mark.parametrize(
    "time_ms,calls,expected_time_ms,expected_calls",
    [(None, None, None, None), ("", "", None, None), ("0", "0", 0.0, 0), ("2.75", "4", 2.75, 4)],
)
@pytest.mark.parametrize("prefix", ["qdrant_deepcopy", "qdrant_query_deepcopy"])
def test_coerce_qdrant_deepcopy_diagnostics(
    time_ms: object,
    calls: object,
    expected_time_ms: float | None,
    expected_calls: int | None,
    prefix: str,
) -> None:
    row = _row("mem0_qdrant", 10.0)
    row.update({f"{prefix}_time_ms": time_ms, f"{prefix}_calls": calls})

    coerced = _coerce_row(row)

    assert coerced[f"{prefix}_time_ms"] == expected_time_ms
    assert coerced[f"{prefix}_calls"] == expected_calls
    if expected_time_ms is not None:
        assert isinstance(coerced[f"{prefix}_time_ms"], float)
    if expected_calls is not None:
        assert isinstance(coerced[f"{prefix}_calls"], int)


@pytest.mark.parametrize("query_ms,query_calls", [(None, None), (0.0, 0), (0.75, 3)])
def test_qdrant_deepcopy_diagnostics_round_trip_without_changing_timing(
    tmp_path: Path, query_ms: float | None, query_calls: int | None,
) -> None:
    config = _config(tmp_path)
    qdrant = build_result_row(
        config,
        10,
        condition="mem0_qdrant",
        vector_backend="qdrant",
        generation_time_s=2.0,
        retrieval_time_s=0.5,
        vector_search_time_s=0.3,
        qdrant_deepcopy_time_ms=125.5,
        qdrant_deepcopy_calls=12,
        qdrant_query_deepcopy_time_ms=query_ms,
        qdrant_query_deepcopy_calls=query_calls,
        total_input_tokens=100,
        total_output_tokens=40,
    )
    write_reports(config, [qdrant, _row("mem0_jasper", 12.0)], system_metadata={})

    with (config.run_dir / "throughput_merged.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["qdrant_deepcopy_time_ms"] == "125.5"
    assert saved[0]["qdrant_deepcopy_calls"] == "12"
    assert saved[1]["qdrant_deepcopy_time_ms"] == ""
    assert saved[1]["qdrant_deepcopy_calls"] == ""
    assert saved[0]["qdrant_query_deepcopy_time_ms"] == ("" if query_ms is None else str(query_ms))
    assert saved[0]["qdrant_query_deepcopy_calls"] == ("" if query_calls is None else str(query_calls))
    assert saved[1]["qdrant_query_deepcopy_time_ms"] == ""
    assert saved[1]["qdrant_query_deepcopy_calls"] == ""

    restored, jasper = read_existing_results(config.run_dir)
    assert restored == qdrant
    assert restored["throughput_qps"] == 10.0
    assert restored["avg_latency_ms"] == 100.0
    assert restored["retrieval_time_s"] == 0.5
    assert restored["vector_search_time_s"] == 0.3
    assert jasper["qdrant_deepcopy_time_ms"] is None
    assert jasper["qdrant_deepcopy_calls"] is None
    assert jasper["qdrant_query_deepcopy_time_ms"] is None
    assert jasper["qdrant_query_deepcopy_calls"] is None
    assert RESULT_COLUMNS[-2:] == ("qdrant_query_deepcopy_time_ms", "qdrant_query_deepcopy_calls")


def test_legacy_csv_without_deepcopy_diagnostics_can_resume_and_merge(tmp_path: Path) -> None:
    config = _config(tmp_path)
    legacy = _row("mem0_qdrant", 10.0)
    del legacy["qdrant_deepcopy_time_ms"]
    del legacy["qdrant_deepcopy_calls"]
    del legacy["qdrant_query_deepcopy_time_ms"]
    del legacy["qdrant_query_deepcopy_calls"]
    path = condition_csv_path(config.run_dir, "mem0_qdrant")
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(legacy))
        writer.writeheader()
        writer.writerow(legacy)

    merged = merge_result_rows(read_existing_results(config.run_dir), [_row("mem0_jasper", 12.0)])

    assert len(merged) == 2
    assert merged[0]["qdrant_deepcopy_time_ms"] is None
    assert merged[0]["qdrant_deepcopy_calls"] is None
    assert merged[0]["qdrant_query_deepcopy_time_ms"] is None
    assert merged[0]["qdrant_query_deepcopy_calls"] is None
