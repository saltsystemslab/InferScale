from __future__ import annotations

import csv
import json
from pathlib import Path

from locomo_jasper_bench.throughput.config import BenchmarkPoint, ThroughputConfig
from locomo_jasper_bench.throughput.reporting import RESULT_COLUMNS, merge_result_rows, write_reports


def _config(tmp_path: Path) -> ThroughputConfig:
    return ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
        run_id="report-test",
        conditions=("prompt_injection", "kv_injection"),
        matrix=(BenchmarkPoint(10, 512),),
    )


def _row(condition: str, qps: float) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": "report-test",
        "model": "test/model",
        "model_label": "test",
        "condition": condition,
        "vector_backend": None,
        "jasper_effective_beam_width": None,
        "num_users": 10,
        "memory_tokens": 512,
        "requests_per_user": 2,
        "total_requests": 20,
        "wall_time_s": 20 / qps,
        "throughput_qps": qps,
        "avg_latency_ms": 1000 / qps,
        "generation_time_s": 20 / qps,
        "retrieval_time_s": 0.0,
        "vector_search_time_s": 0.0,
        "prompt_build_time_s": 0.1,
        "memory_setup_time_s": 0.0,
        "kv_precompute_time_s": 1.0 if condition == "kv_injection" else 0.0,
        "engine_startup_time_s": 2.0,
        "kv_store_gpu_mb": 10.0 if condition == "kv_injection" else 0.0,
        "total_input_tokens": 10240,
        "total_output_tokens": 1000,
        "input_tokens_per_second": 5120.0,
        "output_tokens_per_second": 500.0,
    }
    assert tuple(values) == RESULT_COLUMNS
    return values


def test_merge_result_rows_replaces_same_condition_and_point() -> None:
    merged = merge_result_rows([_row("kv_injection", 10.0)], [_row("kv_injection", 12.0)])

    assert len(merged) == 1
    assert merged[0]["throughput_qps"] == 12.0


def test_write_reports_creates_csv_json_and_markdown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = write_reports(
        config,
        [_row("prompt_injection", 10.0), _row("kv_injection", 20.0)],
        system_metadata={"gpu": {"available": False}},
    )

    assert summary["row_count"] == 2
    with (config.run_dir / "throughput_merged.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    saved_summary = json.loads((config.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["conditions"]["kv_injection"]["maximum_qps"] == 20.0
    report = (config.run_dir / "throughput_report.md").read_text(encoding="utf-8")
    assert "2.00x" in report
