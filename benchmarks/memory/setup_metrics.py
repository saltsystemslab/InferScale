"""Per-sample setup metrics: one row per conversation describing how its memory was built."""

from __future__ import annotations

from typing import Any

from benchmarks.common.stats import as_number
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample
from benchmarks.memory.modes import result_mode

SAMPLE_SETUP_COLUMNS = [
    "run_id",
    "mode",
    "sample_id",
    "question_count",
    "turn_count",
    "vector_backend",
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
    "llama_kv_chunk_metadata_cpu_bytes",
    "llama_kv_chunk_metadata_cpu_mb",
    "llama_kv_chunk_map_gpu_bytes",
    "llama_kv_chunk_map_gpu_mb",
    "llama_kv_chunk_tensor_gpu_bytes",
    "llama_kv_chunk_tensor_gpu_mb",
    "llama_kv_prefix_tensor_gpu_bytes",
    "llama_kv_prefix_tensor_gpu_mb",
    "llama_kv_total_tensor_gpu_bytes",
    "llama_kv_total_tensor_gpu_mb",
    "answer_prepare_sample_time_ms",
    "sample_setup_time_ms",
]

SETUP_TEXT_COLUMNS = ("run_id", "mode", "sample_id", "vector_backend")
SETUP_NUMERIC_COLUMNS = tuple(
    column for column in SAMPLE_SETUP_COLUMNS if column not in SETUP_TEXT_COLUMNS
)
# Summarized in summary.json; the counts describe the sample, not its setup.
SETUP_METRIC_COLUMNS = tuple(
    column for column in SETUP_NUMERIC_COLUMNS if column not in ("question_count", "turn_count")
)
# Copied from the KV precompute metrics into the setup row.
KV_PRECOMPUTE_SETUP_KEYS = tuple(
    column
    for column in SAMPLE_SETUP_COLUMNS
    if column.startswith(("kv_precomputed_", "kv_chunk_cache_", "llama_kv_"))
)


def empty_setup_row(
    config: MemoryRunConfig, sample: ConversationSample, question_count: int
) -> dict[str, Any]:
    """A setup row with every column present; metrics are filled in as setup proceeds."""
    row: dict[str, Any] = dict.fromkeys(SAMPLE_SETUP_COLUMNS)
    row.update(
        run_id=config.run_id,
        mode=result_mode(config),
        sample_id=sample.sample_id,
        question_count=question_count,
        turn_count=len(sample.turns),
        vector_backend=config.vector_backend,
    )
    return row


def record_kv_precompute(row: dict[str, Any], kv_metrics: dict[str, Any]) -> None:
    row["kv_precompute_time_ms"] = as_number(kv_metrics.get("kv_precompute_time_ms"))
    for key in KV_PRECOMPUTE_SETUP_KEYS:
        row[key] = as_number(kv_metrics.get(key))


def setup_total_ms(row: dict[str, Any]) -> float:
    parts = (
        as_number(row.get("memory_setup_time_ms")),
        as_number(row.get("kv_precompute_time_ms")),
        as_number(row.get("answer_prepare_sample_time_ms")),
    )
    return sum(part for part in parts if part is not None)
