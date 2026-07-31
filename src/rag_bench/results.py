from __future__ import annotations

from typing import Any, Iterable

from locomo_jasper_bench.results import percentile

QUESTION_TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")

# Question types whose gold behavior is abstention: MultiHop-RAG null queries
# and QASPER questions whose first reference is unanswerable.
ABSTENTION_QUESTION_TYPES = frozenset({"null_query", "unanswerable"})

ANSWER_METRIC_KEYS = ("exact_match", "f1", "substring_match")

RETRIEVAL_METRIC_KEYS = (
    "evidence_recall_at_k",
    "evidence_full_recall_at_k",
    "evidence_hit_any_at_k",
    "doc_mrr_at_k",
    "fact_recall_at_k",
)

TIMING_METRIC_KEYS = (
    "time_to_first_token_ms",
    "query_to_first_token_ms",
    "query_to_answer_ms",
    "query_embedding_time_ms",
    "vector_db_query_time_ms",
    "query_retrieval_time_ms",
    "kv_memory_tokens",
    "kv_query_tokens",
    "kv_fetch_time_ms",
    "kv_compose_time_ms",
    "kv_verify_time_ms",
    "kv_store_write_time_ms",
    "kv_loaded_memory_tokens",
    "kv_recomputed_memory_tail_tokens",
    "kv_engine_time_to_first_token_ms",
    "prefix_engine_time_to_first_token_ms",
    "answer_generate_time_ms",
    "answer_total_time_ms",
    "answer_time_to_first_token_ms",
)

QUERY_METRICS_COLUMNS = [
    "query_id",
    "question_type",
    "exact_match",
    "f1",
    "substring_match",
    "predicted_insufficient",
    "judge_correct",
    "evidence_recall_at_k",
    "evidence_full_recall_at_k",
    "evidence_hit_any_at_k",
    "doc_mrr_at_k",
    "fact_recall_at_k",
    "retrieved_chunk_count",
    "retrieved_doc_count",
    "kv_memory_tokens",
    "kv_fetch_time_ms",
    "kv_compose_time_ms",
    "kv_verify_time_ms",
    "answer_generate_time_ms",
    "answer_total_time_ms",
    "answer_time_to_first_token_ms",
    "query_to_answer_ms",
    "query_embedding_time_ms",
    "vector_db_query_time_ms",
    "query_retrieval_time_ms",
]


def summarize_rag_records(
    records: Iterable[dict[str, Any]],
    *,
    run_id: str,
    mode: str,
    config: dict[str, Any],
    system_metadata: dict[str, Any],
    setup_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(records)
    judged = [row for row in rows if isinstance(row.get("judge", {}).get("correct"), bool)]
    correct = sum(1 for row in judged if row["judge"]["correct"] is True)
    abstention_rows = [
        row for row in rows if row.get("question_type") in ABSTENTION_QUESTION_TYPES
    ]
    answerable = [
        row for row in rows if row.get("question_type") not in ABSTENTION_QUESTION_TYPES
    ]
    retrieval_rows = [row for row in rows if isinstance(row.get("retrieval"), dict)]

    metrics: dict[str, Any] = {
        "accuracy": _safe_div(correct, len(judged)),
        "accuracy_by_type": _accuracy_by_type(judged),
        "question_type_counts": _type_counts(rows),
        "abstention_accuracy": _mean(
            _answer_metric_values(abstention_rows, "predicted_insufficient")
        ),
        "false_abstention_rate": _mean(
            _answer_metric_values(answerable, "predicted_insufficient")
        ),
    }
    for key in ANSWER_METRIC_KEYS:
        metrics[key] = _mean(_answer_metric_values(rows, key))
    metrics["answer_metrics_by_type"] = {
        question_type: {
            key: _mean(_answer_metric_values(type_rows, key)) for key in ANSWER_METRIC_KEYS
        }
        for question_type, type_rows in _rows_by_type(rows).items()
    }
    metrics["retrieval"] = {
        "query_count": len(retrieval_rows),
        **{
            key: _mean(_retrieval_metric_values(retrieval_rows, key))
            for key in RETRIEVAL_METRIC_KEYS
        },
    }
    for key in TIMING_METRIC_KEYS:
        summary = _numeric_summary(_timing_metric_values(rows, key))
        if summary["count"]:
            metrics[key] = summary

    return {
        "run_id": run_id,
        "mode": mode,
        "question_count": len(rows),
        "judged_count": len(judged),
        "correct_count": correct,
        "metrics": metrics,
        "setup": dict(setup_metrics or {}),
        "config": config,
        "system": system_metadata,
    }


def build_query_metric_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        answer = record.get("answer_metrics") or {}
        retrieval = record.get("retrieval") or {}
        metrics = record.get("metrics") or {}
        judge = record.get("judge") or {}
        judge_correct = judge.get("correct")
        rows.append(
            {
                "query_id": record.get("query_id"),
                "question_type": record.get("question_type"),
                "exact_match": _as_int(answer.get("exact_match")),
                "f1": answer.get("f1"),
                "substring_match": _as_int(answer.get("substring_match")),
                "predicted_insufficient": _as_int(answer.get("predicted_insufficient")),
                "judge_correct": _as_int(judge_correct) if isinstance(judge_correct, bool) else None,
                "evidence_recall_at_k": retrieval.get("evidence_recall_at_k"),
                "evidence_full_recall_at_k": retrieval.get("evidence_full_recall_at_k"),
                "evidence_hit_any_at_k": retrieval.get("evidence_hit_any_at_k"),
                "doc_mrr_at_k": retrieval.get("doc_mrr_at_k"),
                "fact_recall_at_k": retrieval.get("fact_recall_at_k"),
                "retrieved_chunk_count": retrieval.get("retrieved_chunk_count")
                or metrics.get("retrieved_chunk_count"),
                "retrieved_doc_count": retrieval.get("retrieved_doc_count"),
                "kv_memory_tokens": metrics.get("kv_memory_tokens"),
                "kv_fetch_time_ms": metrics.get("kv_fetch_time_ms"),
                "kv_compose_time_ms": metrics.get("kv_compose_time_ms"),
                "kv_verify_time_ms": metrics.get("kv_verify_time_ms"),
                "answer_generate_time_ms": metrics.get("answer_generate_time_ms"),
                "answer_total_time_ms": metrics.get("answer_total_time_ms"),
                "answer_time_to_first_token_ms": metrics.get("answer_time_to_first_token_ms"),
                "query_to_answer_ms": metrics.get("query_to_answer_ms"),
                "query_embedding_time_ms": metrics.get("query_embedding_time_ms"),
                "vector_db_query_time_ms": metrics.get("vector_db_query_time_ms"),
                "query_retrieval_time_ms": metrics.get("query_retrieval_time_ms"),
            }
        )
    return rows


def _rows_by_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("question_type") or "unknown"), []).append(row)
    return dict(sorted(grouped.items()))


def _accuracy_by_type(judged: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        question_type: {
            "total": len(type_rows),
            "correct": sum(1 for row in type_rows if row["judge"]["correct"] is True),
            "accuracy": _safe_div(
                sum(1 for row in type_rows if row["judge"]["correct"] is True),
                len(type_rows),
            ),
        }
        for question_type, type_rows in _rows_by_type(judged).items()
    }


def _type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        question_type: len(type_rows)
        for question_type, type_rows in _rows_by_type(rows).items()
    }


def _answer_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        answer = row.get("answer_metrics")
        if isinstance(answer, dict) and key in answer and answer[key] is not None:
            values.append(float(answer[key]))
    return values


def _retrieval_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        retrieval = row.get("retrieval")
        if isinstance(retrieval, dict) and retrieval.get(key) is not None:
            values.append(float(retrieval[key]))
    return values


def _timing_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and metrics.get(key) is not None:
            try:
                values.append(float(metrics[key]))
            except (TypeError, ValueError):
                continue
    return values


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    numbers = sorted(values)
    if not numbers:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(numbers),
        "avg": sum(numbers) / len(numbers),
        "min": numbers[0],
        "p50": percentile(numbers, 0.50),
        "p95": percentile(numbers, 0.95),
        "max": numbers[-1],
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))
