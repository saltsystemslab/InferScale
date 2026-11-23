from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks.common.files import write_csv
from benchmarks.common.stats import as_number
from benchmarks.memory.plots import write_token_plots
from benchmarks.memory.setup_metrics import SAMPLE_SETUP_COLUMNS, SETUP_NUMERIC_COLUMNS

QUERY_METRICS_FILE = "query_metrics.csv"
SAMPLE_SETUP_FILE = "sample_setup_metrics.csv"
PLOTS_FOLDER = "plots"
ACCURACY_BINS_FILE = "tokens_vs_accuracy_binned.csv"
Z_SCORE_95 = 1.96

QUERY_METRICS_COLUMNS = [
    "run_id",
    "mode",
    "sample_id",
    "question_id",
    "category",
    "resolved_vector_backend",
    "jasper_effective_beam_width",
    "memory_tokens",
    "query_tokens",
    "total_prompt_tokens",
    "memory_context_window",
    "memory_retrieved_fact_ids",
    "memory_retrieved_fact_text_hashes",
    "memory_injected_fact_ids",
    "memory_context_turn_ids",
    "memory_context_turn_count",
    "memory_context_encoding_tokens_total",
    "memory_context_encoding_tokens_max",
    "memory_context_encoding_truncated_tokens",
    "memory_context_text_tokens",
    "memory_token_budget",
    "kv_block_size",
    "kv_loaded_memory_tokens",
    "kv_recomputed_memory_tail_tokens",
    "kv_fact_tokens_end",
    "kv_verify_time_ms",
    "lmcache_expected_matched_tokens",
    "lmcache_matched_tokens",
    "lmcache_load_time_ms",
    "time_to_first_token_ms",
    "query_to_first_token_ms",
    "query_to_answer_ms",
    "judge_correct",
    "retrieved_count",
]

RECORD_COLUMNS = ("run_id", "mode", "sample_id", "question_id", "category")
NUMERIC_METRIC_COLUMNS = (
    "jasper_effective_beam_width",
    "memory_context_turn_count",
    "memory_context_encoding_tokens_total",
    "memory_context_encoding_tokens_max",
    "memory_context_encoding_truncated_tokens",
    "memory_context_text_tokens",
    "memory_token_budget",
    "kv_block_size",
    "kv_loaded_memory_tokens",
    "kv_recomputed_memory_tail_tokens",
    "kv_fact_tokens_end",
    "kv_verify_time_ms",
    "lmcache_expected_matched_tokens",
    "lmcache_matched_tokens",
    "lmcache_load_time_ms",
    "time_to_first_token_ms",
    "query_to_first_token_ms",
    "query_to_answer_ms",
)
LIST_METRIC_COLUMNS = (
    "memory_retrieved_fact_ids",
    "memory_retrieved_fact_text_hashes",
    "memory_injected_fact_ids",
    "memory_context_turn_ids",
)

ACCURACY_BIN_COLUMNS = [
    "token_bin_start",
    "token_bin_end",
    "token_bin_midpoint",
    "judged_count",
    "correct_count",
    "accuracy",
    "ci_lower",
    "ci_upper",
]


def write_query_reports(run_dir: str | Path, records: Iterable[dict[str, Any]]) -> None:
    run_path = Path(run_dir)
    rows = query_metric_rows(records)
    write_csv(run_path / QUERY_METRICS_FILE, rows, QUERY_METRICS_COLUMNS)
    plots_dir = run_path / PLOTS_FOLDER
    accuracy_bins = accuracy_bin_rows(rows)
    write_csv(plots_dir / ACCURACY_BINS_FILE, accuracy_bins, ACCURACY_BIN_COLUMNS)
    write_token_plots(plots_dir, rows, accuracy_bins)


def write_sample_setup_report(run_dir: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_csv(Path(run_dir) / SAMPLE_SETUP_FILE, rows, SAMPLE_SETUP_COLUMNS)


def read_sample_setup_report(run_dir: str | Path) -> list[dict[str, Any]]:
    report_path = Path(run_dir) / SAMPLE_SETUP_FILE
    if not report_path.exists():
        return []
    with report_path.open(newline="", encoding="utf-8") as report:
        return [_setup_row_from_csv(row) for row in csv.DictReader(report)]


def query_metric_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_query_metric_row(record) for record in records]


def accuracy_bin_rows(
    rows: Iterable[dict[str, Any]], *, max_bins: int = 10
) -> list[dict[str, Any]]:
    judged = [row for row in rows if _has_token_count(row) and _is_judged(row)]
    judged.sort(key=lambda row: float(row["total_prompt_tokens"]))
    if not judged:
        return []
    bin_count = min(max_bins, max(1, round(math.sqrt(len(judged)))))
    rows_per_bin = max(1, math.ceil(len(judged) / bin_count))
    return [
        _accuracy_bin(judged[start : start + rows_per_bin])
        for start in range(0, len(judged), rows_per_bin)
    ]


def judged_rows_missing_tokens(rows: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in rows if _is_judged(row) and not _has_token_count(row))


def _is_judged(row: dict[str, Any]) -> bool:
    return row.get("judge_correct") in {0, 1}


def _has_token_count(row: dict[str, Any]) -> bool:
    return as_number(row.get("total_prompt_tokens")) is not None


def _query_metric_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = _metrics(record)
    memory_tokens = as_number(metrics.get("kv_memory_tokens"))
    query_tokens = as_number(metrics.get("kv_query_tokens"))
    retrieved_memories = record.get("retrieved_memories")
    cells: dict[str, Any] = {column: record.get(column) for column in RECORD_COLUMNS}
    cells["resolved_vector_backend"] = metrics.get("resolved_vector_backend")
    cells["memory_tokens"] = memory_tokens
    cells["query_tokens"] = query_tokens
    cells["total_prompt_tokens"] = (
        memory_tokens + query_tokens
        if memory_tokens is not None and query_tokens is not None
        else None
    )
    cells["memory_context_window"] = as_number(_context_window(metrics))
    for column in NUMERIC_METRIC_COLUMNS:
        cells[column] = as_number(metrics.get(column))
    for column in LIST_METRIC_COLUMNS:
        metric = metrics.get(column)
        cells[column] = metric if isinstance(metric, list) else None
    cells["judge_correct"] = _judge_correct(record.get("judge"))
    cells["retrieved_count"] = (
        len(retrieved_memories) if isinstance(retrieved_memories, list) else None
    )
    return {column: cells[column] for column in QUERY_METRICS_COLUMNS}


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _context_window(metrics: dict[str, Any]) -> Any:
    context_window = metrics.get("memory_context_window")
    if context_window is None:
        return metrics.get("kv_context_window")
    return context_window


def _judge_correct(judge: Any) -> int | None:
    if not isinstance(judge, dict):
        return None
    correct = judge.get("correct")
    if correct is True:
        return 1
    if correct is False:
        return 0
    return None


def _setup_row_from_csv(row: dict[str, str]) -> dict[str, Any]:
    return {column: _setup_cell(column, text) for column, text in row.items()}


def _setup_cell(column: str, text: str) -> Any:
    if column not in SETUP_NUMERIC_COLUMNS:
        return text
    if text == "":
        return None
    return as_number(text)


def _accuracy_bin(bin_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [float(row["total_prompt_tokens"]) for row in bin_rows]
    correct_count = sum(int(row["judge_correct"]) for row in bin_rows)
    ci_lower, ci_upper = _wilson_interval(correct_count, len(bin_rows))
    return {
        "token_bin_start": min(tokens),
        "token_bin_end": max(tokens),
        "token_bin_midpoint": (min(tokens) + max(tokens)) / 2,
        "judged_count": len(bin_rows),
        "correct_count": correct_count,
        "accuracy": correct_count / len(bin_rows),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _wilson_interval(correct: int, total: int, *, z: float = Z_SCORE_95) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    accuracy = correct / total
    denominator = 1 + z**2 / total
    center = (accuracy + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((accuracy * (1 - accuracy) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
