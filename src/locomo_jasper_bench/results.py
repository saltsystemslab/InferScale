from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable


# Upstream memory-benchmarks LoCoMo category names; 5 (adversarial) is excluded
# from runs but kept here so historical records still summarize cleanly.
CATEGORY_NAMES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}

SETUP_METRIC_KEYS = (
    "memory_create_time_ms",
    "embedding_memory_build_time_ms",
    "memory_input_turn_count",
    "memory_inferred_record_count",
    "memory_fact_catalog_loaded",
    "memory_llm_cache_hits",
    "memory_llm_cache_misses",
    "vector_index_build_time_ms",
    "jasper_vector_count",
    "jasper_embedding_dim",
    "jasper_embedding_matrix_cpu_bytes",
    "jasper_embedding_matrix_cpu_mb",
    "jasper_embedding_matrix_gpu_logical_bytes",
    "jasper_embedding_matrix_gpu_logical_mb",
    "jasper_graph_gpu_bytes",
    "jasper_graph_gpu_mb",
    "jasper_graph_torch_allocated_delta_bytes",
    "jasper_graph_torch_allocated_delta_mb",
    "memory_setup_time_ms",
    "kv_precompute_time_ms",
    "kv_precomputed_chunks",
    "kv_precomputed_chunks_with_prefix",
    "kv_precomputed_tokens",
    "kv_precomputed_layers",
    "kv_precomputed_gpu_mb",
    "kv_chunk_cache_residency_is_gpu",
    "llama_kv_chunk_count",
    "llama_kv_chunk_map_cpu_bytes",
    "llama_kv_chunk_map_cpu_mb",
    "llama_kv_chunk_tensor_gpu_bytes",
    "llama_kv_chunk_tensor_gpu_mb",
    "llama_kv_prefix_tensor_gpu_bytes",
    "llama_kv_prefix_tensor_gpu_mb",
    "llama_kv_total_tensor_gpu_bytes",
    "llama_kv_total_tensor_gpu_mb",
    "answer_prepare_sample_time_ms",
    "sample_setup_time_ms",
)

MEMORY_AUDIT_NUMERIC_METRIC_KEYS = (
    "memory_context_window",
    "memory_context_turn_count",
    "memory_context_encoding_tokens_total",
    "memory_context_encoding_tokens_max",
    "memory_context_encoding_truncated_tokens",
    "memory_context_text_tokens",
    "memory_token_budget",
)

LEGACY_KV_QUERY_METRIC_KEYS = (
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
    "prefix_engine_time_to_first_token_ms",
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
    judged = [row for row in rows if row.get("judge", {}).get("correct") is not None]
    correct = sum(1 for row in judged if row.get("judge", {}).get("correct") is True)
    accuracy_by_category = _accuracy_by_category(judged)
    vector_query_times = _number_values(_metric_values(rows, "vector_db_query_time_ms"))
    vector_query_total_ms = sum(vector_query_times)
    vector_query_count = len(vector_query_times)

    metrics = {
        "accuracy": _safe_div(correct, len(judged)),
        "accuracy_by_category": accuracy_by_category,
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
    for key in (*MEMORY_AUDIT_NUMERIC_METRIC_KEYS, *LEGACY_KV_QUERY_METRIC_KEYS):
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
        "judged_count": len(judged),
        "correct_count": correct,
        "metrics": metrics,
        "config": config,
        "system": system_metadata,
    }


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
            "accuracy": _safe_div(counters["correct"], counters["total"]),
        }
        for category, counters in sorted(by_category.items())
    }


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
    return [float(value) for value in values if value is not None]


def _numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    numbers = sorted(_number_values(values))
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


def percentile(sorted_numbers: list[float], fraction: float) -> float:
    if len(sorted_numbers) == 1:
        return sorted_numbers[0]
    index = (len(sorted_numbers) - 1) * fraction
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
