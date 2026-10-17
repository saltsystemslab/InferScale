"""The summary.json and query_metrics.csv of a RAG run."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from benchmarks.common.run_summary import correct_count, judged_records, summary_envelope
from benchmarks.common.stats import mean, numeric_summary, safe_div
from benchmarks.rag.data_types import ABSTENTION_QUESTION_TYPES

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
    "prompt_injection_engine_time_to_first_token_ms",
    "answer_generate_time_ms",
    "answer_total_time_ms",
    "answer_time_to_first_token_ms",
)

Record = dict[str, Any]
ColumnReader = Callable[[Record], Any]


def _section(record: Record, name: str) -> dict[str, Any]:
    value = record.get(name)
    return value if isinstance(value, dict) else {}


def _field(key: str) -> ColumnReader:
    return lambda record: record.get(key)


def _value(section: str, key: str) -> ColumnReader:
    return lambda record: _section(record, section).get(key)


def _flag(section: str, key: str) -> ColumnReader:
    return lambda record: _as_flag(_section(record, section).get(key))


def _judge_flag(record: Record) -> int | None:
    verdict = _section(record, "judge").get("correct")
    return int(verdict) if isinstance(verdict, bool) else None


def _retrieved_chunk_count(record: Record) -> Any:
    retrieval = _section(record, "retrieval").get("retrieved_chunk_count")
    return retrieval or _section(record, "metrics").get("retrieved_chunk_count")


def _as_flag(value: Any) -> int | None:
    return None if value is None else int(bool(value))


# The CSV columns, in order, and where each one is read from in a prediction record.
QUERY_METRIC_COLUMNS: tuple[tuple[str, ColumnReader], ...] = (
    ("query_id", _field("query_id")),
    ("question_type", _field("question_type")),
    ("exact_match", _flag("answer_metrics", "exact_match")),
    ("f1", _value("answer_metrics", "f1")),
    ("substring_match", _flag("answer_metrics", "substring_match")),
    ("predicted_insufficient", _flag("answer_metrics", "predicted_insufficient")),
    ("judge_correct", _judge_flag),
    ("evidence_recall_at_k", _value("retrieval", "evidence_recall_at_k")),
    ("evidence_full_recall_at_k", _value("retrieval", "evidence_full_recall_at_k")),
    ("evidence_hit_any_at_k", _value("retrieval", "evidence_hit_any_at_k")),
    ("doc_mrr_at_k", _value("retrieval", "doc_mrr_at_k")),
    ("fact_recall_at_k", _value("retrieval", "fact_recall_at_k")),
    ("retrieved_chunk_count", _retrieved_chunk_count),
    ("retrieved_doc_count", _value("retrieval", "retrieved_doc_count")),
    ("kv_memory_tokens", _value("metrics", "kv_memory_tokens")),
    ("kv_fetch_time_ms", _value("metrics", "kv_fetch_time_ms")),
    ("kv_compose_time_ms", _value("metrics", "kv_compose_time_ms")),
    ("kv_verify_time_ms", _value("metrics", "kv_verify_time_ms")),
    ("answer_generate_time_ms", _value("metrics", "answer_generate_time_ms")),
    ("answer_total_time_ms", _value("metrics", "answer_total_time_ms")),
    ("answer_time_to_first_token_ms", _value("metrics", "answer_time_to_first_token_ms")),
    ("query_to_answer_ms", _value("metrics", "query_to_answer_ms")),
    ("query_embedding_time_ms", _value("metrics", "query_embedding_time_ms")),
    ("vector_db_query_time_ms", _value("metrics", "vector_db_query_time_ms")),
    ("query_retrieval_time_ms", _value("metrics", "query_retrieval_time_ms")),
)
QUERY_METRICS_COLUMNS = [name for name, _ in QUERY_METRIC_COLUMNS]


def build_query_metric_rows(records: Iterable[Record]) -> list[Record]:
    return [{name: read(record) for name, read in QUERY_METRIC_COLUMNS} for record in records]


def summarize_rag_records(
    records: Iterable[Record],
    *,
    run_id: str,
    mode: str,
    config: dict[str, Any],
    system_metadata: dict[str, Any],
    setup_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(records)
    judged = judged_records(rows)
    by_type = _rows_by_type(rows)
    abstaining = [row for row in rows if row.get("question_type") in ABSTENTION_QUESTION_TYPES]
    answerable = [row for row in rows if row.get("question_type") not in ABSTENTION_QUESTION_TYPES]
    retrieval_rows = [row for row in rows if isinstance(row.get("retrieval"), dict)]

    metrics: dict[str, Any] = {
        "accuracy": safe_div(correct_count(judged), len(judged)),
        "accuracy_by_type": _accuracy_by_type(judged),
        "question_type_counts": {name: len(type_rows) for name, type_rows in by_type.items()},
        "abstention_accuracy": mean(
            _section_values(abstaining, "answer_metrics", "predicted_insufficient")
        ),
        "false_abstention_rate": mean(
            _section_values(answerable, "answer_metrics", "predicted_insufficient")
        ),
    }
    for key in ANSWER_METRIC_KEYS:
        metrics[key] = mean(_section_values(rows, "answer_metrics", key))
    metrics["answer_metrics_by_type"] = {
        name: {
            key: mean(_section_values(type_rows, "answer_metrics", key))
            for key in ANSWER_METRIC_KEYS
        }
        for name, type_rows in by_type.items()
    }
    metrics["retrieval"] = {
        "query_count": len(retrieval_rows),
        **{
            key: mean(_section_values(retrieval_rows, "retrieval", key))
            for key in RETRIEVAL_METRIC_KEYS
        },
    }
    for key in TIMING_METRIC_KEYS:
        summary = numeric_summary(_section_values(rows, "metrics", key))
        if summary["count"]:
            metrics[key] = summary
    return summary_envelope(
        rows,
        run_id=run_id,
        mode=mode,
        metrics=metrics,
        config=config,
        system_metadata=system_metadata,
        setup=dict(setup_metrics or {}),
    )


def _rows_by_type(rows: list[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("question_type") or "unknown"), []).append(row)
    return dict(sorted(grouped.items()))


def _accuracy_by_type(judged: list[Record]) -> dict[str, dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for name, type_rows in _rows_by_type(judged).items():
        correct = correct_count(type_rows)
        by_type[name] = {
            "total": len(type_rows),
            "correct": correct,
            "accuracy": safe_div(correct, len(type_rows)),
        }
    return by_type


def _section_values(rows: list[Record], section: str, key: str) -> list[float]:
    """The numeric values of ``key`` in ``section`` across rows; booleans count as 0 or 1."""
    values: list[float] = []
    for row in rows:
        value = _section(row, section).get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values
