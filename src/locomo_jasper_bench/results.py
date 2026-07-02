from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable

from .data import is_adversarial_category


SETUP_METRIC_KEYS = (
    "memory_create_time_ms",
    "embedding_memory_build_time_ms",
    "vector_index_build_time_ms",
    "memory_setup_time_ms",
    "kv_precompute_time_ms",
    "answer_prepare_sample_time_ms",
    "sample_setup_time_ms",
)


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
    sample_setup_metrics: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = list(records)
    setup_rows = list(sample_setup_metrics or [])
    judged = [row for row in rows if _judge_verdict(row) is not None]
    # Headline accuracy follows the LoCoMo convention: adversarial (category-5)
    # questions are scored separately and excluded from the overall number.
    scored = [row for row in judged if not is_adversarial_category(row.get("category") or "")]
    scored_correct = sum(1 for row in scored if _judge_verdict(row) is True)
    adversarial_rows = [row for row in rows if is_adversarial_category(row.get("category") or "")]
    adversarial_judged = [row for row in adversarial_rows if _judge_verdict(row) is not None]
    adversarial_correct = sum(1 for row in adversarial_judged if _judge_verdict(row) is True)
    total_correct = sum(1 for row in judged if _judge_verdict(row) is True)
    vector_query_times = _number_values(_metric_values(rows, "vector_db_query_time_ms"))
    vector_query_total_ms = sum(vector_query_times)
    vector_query_count = len(vector_query_times)

    metrics = {
        "accuracy": _safe_div(scored_correct, len(scored)),
        "accuracy_by_category": _accuracy_by_category(judged),
        "judge_unparsed_count": _judge_unparsed_count(rows),
        "time_to_first_token_ms": _numeric_summary(_metric_values(rows, "time_to_first_token_ms")),
        "query_to_first_token_ms": _numeric_summary(_metric_values(rows, "query_to_first_token_ms")),
        "query_to_answer_ms": _numeric_summary(_metric_values(rows, "query_to_answer_ms")),
        "query_embedding_time_ms": _numeric_summary(_metric_values(rows, "query_embedding_time_ms")),
        "query_retrieval_time_ms": _numeric_summary(_metric_values(rows, "query_retrieval_time_ms")),
        "vector_db_query_time_ms": _numeric_summary(vector_query_times),
        "vector_db_query_count": vector_query_count,
        "vector_db_query_time_total_ms": vector_query_total_ms,
        "vector_db_queries_per_sec": _queries_per_second(vector_query_count, vector_query_total_ms),
    }
    for key in (
        "kv_memory_tokens",
        "kv_compose_time_ms",
        "answer_generate_time_ms",
        "answer_total_time_ms",
        "answer_time_to_first_token_ms",
        "kv_engine_time_to_first_token_ms",
        "kv_query_tokens",
        "kv_query_bos_stripped",
        "kv_context_window",
        "kv_context_prefix_tokens_total",
        "kv_context_prefix_tokens_max",
        "kv_context_prefix_truncated_tokens",
        "kv_store_gpu_mb",
        "prefix_engine_time_to_first_token_ms",
    ):
        summary = _numeric_summary(_metric_values(rows, key))
        if summary["count"]:
            metrics[key] = summary
    if setup_rows:
        metrics["sample_setup_count"] = len(setup_rows)
        for key in SETUP_METRIC_KEYS:
            summary = _numeric_summary(_setup_metric_values(setup_rows, key))
            if summary["count"]:
                metrics[key] = summary

    return {
        "run_id": run_id,
        "mode": mode,
        "question_count": len(rows),
        "scored_question_count": len(
            [row for row in rows if not is_adversarial_category(row.get("category") or "")]
        ),
        "judged_count": len(scored),
        "correct_count": scored_correct,
        "total_judged_count": len(judged),
        "total_correct_count": total_correct,
        "adversarial_question_count": len(adversarial_rows),
        "adversarial_judged_count": len(adversarial_judged),
        "adversarial_correct_count": adversarial_correct,
        "metrics": metrics,
        "config": config,
        "system": system_metadata,
    }


def _judge_verdict(row: dict[str, Any]) -> bool | None:
    judge = row.get("judge")
    if isinstance(judge, dict):
        verdict = judge.get("correct")
        if isinstance(verdict, bool):
            return verdict
    return None


def _accuracy_by_category(judged_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, int]] = {}
    for row in judged_rows:
        category = str(row.get("category") or "unknown")
        bucket = buckets.setdefault(category, {"judged": 0, "correct": 0})
        bucket["judged"] += 1
        if _judge_verdict(row) is True:
            bucket["correct"] += 1
    return {
        category: {
            "judged": bucket["judged"],
            "correct": bucket["correct"],
            "accuracy": _safe_div(bucket["correct"], bucket["judged"]),
        }
        for category, bucket in sorted(buckets.items())
    }


def _judge_unparsed_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        judge = row.get("judge")
        if not isinstance(judge, dict) or _judge_verdict(row) is not None:
            continue
        if judge.get("status") == "unparsed" or judge.get("raw"):
            count += 1
    return count


def _metric_values(rows: list[dict[str, Any]], key: str) -> Iterable[Any]:
    for row in rows:
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and key in metrics:
            yield metrics.get(key)


def _setup_metric_values(rows: list[dict[str, Any]], key: str) -> Iterable[Any]:
    for row in rows:
        if isinstance(row, dict) and key in row:
            yield row.get(key)


def _number_values(values: Iterable[Any]) -> list[float]:
    return [number for number in (coerce_number(value) for value in values) if number is not None]


def coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _queries_per_second(query_count: int, total_ms: float) -> float | None:
    if query_count == 0 or total_ms <= 0:
        return None
    return query_count / (total_ms / 1000)
