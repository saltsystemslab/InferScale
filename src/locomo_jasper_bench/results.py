from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable


class JsonlWriter(AbstractContextManager["JsonlWriter"]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_records(
    records: Iterable[dict[str, Any]],
    *,
    run_id: str,
    mode: str,
    config: dict[str, Any],
    system_metadata: dict[str, Any],
) -> dict[str, Any]:
    rows = list(records)
    judged = [row for row in rows if _judge_correct(row) is not None]
    correct = sum(1 for row in judged if _judge_correct(row) is True)
    vector_query_times = _number_values(_metric_values(rows, "vector_db_query_time_ms"))
    vector_query_total_ms = sum(vector_query_times)
    vector_query_count = len(vector_query_times)
    metrics = {
        "accuracy": _safe_div(correct, len(judged)),
        "time_to_first_token_ms": _numeric_summary(_metric_values(rows, "time_to_first_token_ms")),
        "vector_db_query_time_ms": _numeric_summary(vector_query_times),
        "vector_db_query_count": vector_query_count,
        "vector_db_query_time_total_ms": vector_query_total_ms,
        "vector_db_queries_per_sec": _queries_per_second(vector_query_count, vector_query_total_ms),
    }
    retrieval_diagnostics = _summarize_retrieval_diagnostics(rows)
    if retrieval_diagnostics is not None:
        metrics["retrieval_diagnostics"] = retrieval_diagnostics

    return {
        "run_id": run_id,
        "mode": mode,
        "question_count": len(rows),
        "judged_count": len(judged),
        "correct_count": correct,
        "metrics": metrics,
        "config": config,
        "system": system_metadata,
    }


def _metric_values(rows: list[dict[str, Any]], key: str) -> Iterable[Any]:
    for row in rows:
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and key in metrics:
            yield metrics.get(key)


def _judge_correct(row: dict[str, Any]) -> Any:
    judge = row.get("judge")
    if not isinstance(judge, dict):
        return None
    return judge.get("correct")


def _number_values(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    numbers = sorted(_number_values(values))
    if not numbers:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(numbers),
        "avg": sum(numbers) / len(numbers),
        "min": numbers[0],
        "p50": _percentile(numbers, 0.50),
        "p95": _percentile(numbers, 0.95),
        "max": numbers[-1],
    }


def _percentile(sorted_numbers: list[float], percentile: float) -> float:
    if len(sorted_numbers) == 1:
        return sorted_numbers[0]
    index = (len(sorted_numbers) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(sorted_numbers) - 1)
    weight = index - lower
    return sorted_numbers[lower] * (1 - weight) + sorted_numbers[upper] * weight


def _summarize_retrieval_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    diagnostics = [
        row.get("retrieval_diagnostics")
        for row in rows
        if isinstance(row.get("retrieval_diagnostics"), dict)
        and row.get("retrieval_diagnostics", {}).get("enabled") is True
    ]
    if not diagnostics:
        return None
    return {
        "count": len(diagnostics),
        "exact_recall_at_requested_top_k": _numeric_summary(
            diagnostic.get("exact_recall_at_requested_top_k") for diagnostic in diagnostics
        ),
        "jasper_candidate_recall_at_diagnostic_k": _numeric_summary(
            diagnostic.get("jasper_candidate_recall_at_diagnostic_k") for diagnostic in diagnostics
        ),
        "exact_top_k_missing_from_retrieved_top_k": _numeric_summary(
            len(diagnostic.get("exact_top_k_missing_from_retrieved_top_k") or [])
            for diagnostic in diagnostics
        ),
        "exact_top_k_found_below_retrieved_top_k": _numeric_summary(
            len(diagnostic.get("exact_top_k_found_below_retrieved_top_k") or [])
            for diagnostic in diagnostics
        ),
        "exact_top_k_missing_from_jasper_candidates": _numeric_summary(
            len(diagnostic.get("exact_top_k_missing_from_jasper_candidates") or [])
            for diagnostic in diagnostics
        ),
    }


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _queries_per_second(query_count: int, total_ms: float) -> float | None:
    if query_count == 0 or total_ms <= 0:
        return None
    return query_count / (total_ms / 1000)
