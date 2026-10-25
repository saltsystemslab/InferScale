"""Per-query and per-sample CSV reports of a memory run."""

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

# Columns copied from the record itself.
RECORD_COLUMNS = ("run_id", "mode", "sample_id", "question_id", "category")
# Columns read from the record's metrics as numbers, under the same key.
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
# Columns read from the record's metrics as lists, under the same key.
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
    """Write per-query token metrics, binned accuracy, and the token-vs-latency plots."""
    run_path = Path(run_dir)
    rows = query_metric_rows(records)
    write_csv(run_path / "query_metrics.csv", rows, QUERY_METRICS_COLUMNS)
    plots_dir = run_path / "plots"
    accuracy_bins = accuracy_bin_rows(rows)
    write_csv(plots_dir / "tokens_vs_accuracy_binned.csv", accuracy_bins, ACCURACY_BIN_COLUMNS)
    write_token_plots(plots_dir, rows, accuracy_bins)


def write_sample_setup_report(run_dir: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_csv(Path(run_dir) / "sample_setup_metrics.csv", rows, SAMPLE_SETUP_COLUMNS)


def read_sample_setup_report(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "sample_setup_metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [_setup_row_from_csv(row) for row in csv.DictReader(handle)]


def query_metric_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_query_metric_row(record) for record in records]


def accuracy_bin_rows(
    rows: Iterable[dict[str, Any]], *, max_bins: int = 10
) -> list[dict[str, Any]]:
    """Judged rows bucketed by prompt tokens into up to ``max_bins`` equal-count bins."""
    judged = [
        row
        for row in rows
        if as_number(row.get("total_prompt_tokens")) is not None
        and row.get("judge_correct") in {0, 1}
    ]
    judged.sort(key=lambda row: float(row["total_prompt_tokens"]))
    if not judged:
        return []
    bin_count = min(max_bins, max(1, round(math.sqrt(len(judged)))))
    chunk_size = max(1, math.ceil(len(judged) / bin_count))
    return [
        _accuracy_bin(judged[start : start + chunk_size])
        for start in range(0, len(judged), chunk_size)
    ]


def judged_rows_missing_tokens(rows: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("judge_correct") in {0, 1} and as_number(row.get("total_prompt_tokens")) is None
    )


def _query_metric_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    memory_tokens = as_number(metrics.get("kv_memory_tokens"))
    query_tokens = as_number(metrics.get("kv_query_tokens"))
    context_window = metrics.get("memory_context_window")
    if context_window is None:
        context_window = metrics.get("kv_context_window")
    retrieved = record.get("retrieved_memories")
    values: dict[str, Any] = {column: record.get(column) for column in RECORD_COLUMNS}
    values["resolved_vector_backend"] = metrics.get("resolved_vector_backend")
    values["memory_tokens"] = memory_tokens
    values["query_tokens"] = query_tokens
    values["total_prompt_tokens"] = (
        memory_tokens + query_tokens
        if memory_tokens is not None and query_tokens is not None
        else None
    )
    values["memory_context_window"] = as_number(context_window)
    for column in NUMERIC_METRIC_COLUMNS:
        values[column] = as_number(metrics.get(column))
    for column in LIST_METRIC_COLUMNS:
        value = metrics.get(column)
        values[column] = value if isinstance(value, list) else None
    values["judge_correct"] = _judge_correct(record.get("judge"))
    values["retrieved_count"] = len(retrieved) if isinstance(retrieved, list) else None
    return {column: values[column] for column in QUERY_METRICS_COLUMNS}


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
    return {
        key: (as_number(value) if value != "" else None) if key in SETUP_NUMERIC_COLUMNS else value
        for key, value in row.items()
    }


def _accuracy_bin(group: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [float(row["total_prompt_tokens"]) for row in group]
    correct_count = sum(int(row["judge_correct"]) for row in group)
    ci_lower, ci_upper = _wilson_interval(correct_count, len(group))
    return {
        "token_bin_start": min(tokens),
        "token_bin_end": max(tokens),
        "token_bin_midpoint": (min(tokens) + max(tokens)) / 2,
        "judged_count": len(group),
        "correct_count": correct_count,
        "accuracy": correct_count / len(group),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _wilson_interval(correct: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
