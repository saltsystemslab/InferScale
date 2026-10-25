"""The summary.json of a memory run: accuracy, latency summaries, and setup summaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from benchmarks.common.run_summary import correct_count, judged_records, summary_envelope
from benchmarks.common.stats import number_values, numeric_summary, queries_per_second, safe_div
from benchmarks.memory.protocol import CATEGORY_NAMES
from benchmarks.memory.setup_metrics import SETUP_METRIC_COLUMNS

MEMORY_AUDIT_NUMERIC_METRIC_KEYS = (
    "memory_context_window",
    "memory_context_turn_count",
    "memory_context_encoding_tokens_total",
    "memory_context_encoding_tokens_max",
    "memory_context_encoding_truncated_tokens",
    "memory_context_text_tokens",
    "memory_token_budget",
)

KV_QUERY_METRIC_KEYS = (
    "kv_memory_tokens",
    "kv_compose_time_ms",
    "kv_verify_time_ms",
    "kv_store_write_time_ms",
    "answer_generate_time_ms",
    "answer_total_time_ms",
    "answer_time_to_first_token_ms",
    "kv_engine_time_to_first_token_ms",
    "kv_query_tokens",
    "kv_query_bos_stripped",
    "kv_context_window",
    "kv_block_size",
    "kv_loaded_memory_tokens",
    "kv_recomputed_memory_tail_tokens",
    "kv_fact_tokens_end",
    "kv_prefix_caching",
    "kv_store_gpu_mb",
    "kv_store_host_mb",
    "kv_h2d_latency_ms",
    "kv_h2d_bytes",
    "kv_h2d_overlap_ratio",
    "kv_staging_stall_ms",
    "jasper_effective_beam_width",
    "prompt_injection_engine_time_to_first_token_ms",
    "lmcache_engine_time_to_first_token_ms",
    "lmcache_expected_matched_tokens",
    "lmcache_matched_tokens",
    "lmcache_load_time_ms",
    "lmcache_separator_tokens",
)


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
    judged = judged_records(rows)
    correct = correct_count(judged)
    vector_query_times = number_values(_metric_values(rows, "vector_db_query_time_ms"))
    vector_query_total_ms = sum(vector_query_times)
    vector_query_count = len(vector_query_times)

    metrics = {
        "accuracy": safe_div(correct, len(judged)),
        "accuracy_by_category": _accuracy_by_category(judged),
        "time_to_first_token_ms": numeric_summary(_metric_values(rows, "time_to_first_token_ms")),
        "query_to_first_token_ms": numeric_summary(_metric_values(rows, "query_to_first_token_ms")),
        "query_to_answer_ms": numeric_summary(_metric_values(rows, "query_to_answer_ms")),
        "query_embedding_time_ms": numeric_summary(_metric_values(rows, "query_embedding_time_ms")),
        "query_retrieval_time_ms": numeric_summary(_metric_values(rows, "query_retrieval_time_ms")),
        "vector_db_query_time_ms": numeric_summary(vector_query_times),
        "vector_db_query_count": vector_query_count,
        "vector_db_query_time_total_ms": vector_query_total_ms,
        "vector_db_queries_per_sec": queries_per_second(vector_query_count, vector_query_total_ms),
    }
    for key in (*MEMORY_AUDIT_NUMERIC_METRIC_KEYS, *KV_QUERY_METRIC_KEYS):
        summary = numeric_summary(_metric_values(rows, key))
        if summary["count"]:
            metrics[key] = summary
    if setup_rows:
        metrics["sample_setup_count"] = len(setup_rows)
        for key in SETUP_METRIC_COLUMNS:
            summary = numeric_summary(row.get(key) for row in setup_rows if key in row)
            if summary["count"]:
                metrics[key] = summary

    return summary_envelope(
        rows,
        run_id=run_id,
        mode=mode,
        metrics=metrics,
        config=config,
        system_metadata=system_metadata,
    )


def _accuracy_by_category(judged: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_category: dict[str, dict[str, int]] = {}
    for row in judged:
        category = str(row.get("category") or "")
        counters = by_category.setdefault(category, {"total": 0, "correct": 0})
        counters["total"] += 1
        if row.get("judge", {}).get("correct") is True:
            counters["correct"] += 1
    return {
        category: {
            "name": CATEGORY_NAMES.get(category, "unknown"),
            "total": counters["total"],
            "correct": counters["correct"],
            "accuracy": safe_div(counters["correct"], counters["total"]),
        }
        for category, counters in sorted(by_category.items())
    }


def _metric_values(rows: list[dict[str, Any]], key: str) -> Iterable[Any]:
    for row in rows:
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and key in metrics:
            yield metrics.get(key)
