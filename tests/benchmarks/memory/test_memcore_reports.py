from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from benchmarks.memory.reporting import (
    ACCURACY_BIN_COLUMNS,
    LIST_METRIC_COLUMNS,
    NUMERIC_METRIC_COLUMNS,
    QUERY_METRICS_COLUMNS,
    query_metric_rows,
    write_query_reports,
)
from benchmarks.memory.setup_metrics import (
    SAMPLE_SETUP_COLUMNS,
    record_kv_precompute,
    record_memory_warmup,
    setup_total_ms,
)


def record(metrics: Any = None, **fields: Any) -> dict[str, Any]:
    return {
        "run_id": "run",
        "mode": "mem0-kv",
        "sample_id": "conv-1",
        "question_id": "q0",
        "category": "1",
        "retrieved_memories": [{"id": "fact-1"}, {"id": "fact-2"}],
        "judge": {"correct": True},
        "metrics": {} if metrics is None else metrics,
        **fields,
    }


def query_row(metrics: Any = None, **fields: Any) -> dict[str, Any]:
    (row,) = query_metric_rows([record(metrics, **fields)])
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as report:
        return list(csv.DictReader(report))


def test_query_reports_are_the_metrics_csv_the_binned_accuracy_csv_and_the_plots(tmp_path):
    records = [
        record({"kv_memory_tokens": 40, "kv_query_tokens": 10, "memory_injected_fact_ids": ["a"]}),
        record({"kv_memory_tokens": 60, "kv_query_tokens": 10}, judge={"correct": False}),
        record({}, question_id="q2"),
    ]

    write_query_reports(tmp_path, records)

    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*.*")) == [
        "plots/tokens_vs_accuracy_binned.csv",
        "plots/tokens_vs_accuracy_binned.png",
        "plots/tokens_vs_query_to_answer.png",
        "plots/tokens_vs_query_to_first_token.png",
        "plots/tokens_vs_ttft.png",
        "query_metrics.csv",
    ]
    query_rows = read_csv(tmp_path / "query_metrics.csv")
    assert [list(row) for row in query_rows] == [QUERY_METRICS_COLUMNS] * 3
    assert [row["total_prompt_tokens"] for row in query_rows] == ["50.0", "70.0", ""]
    assert [row["memory_injected_fact_ids"] for row in query_rows] == ['["a"]', "", ""]
    assert read_csv(tmp_path / "plots" / "tokens_vs_accuracy_binned.csv") == [
        {
            "token_bin_start": "50.0",
            "token_bin_end": "70.0",
            "token_bin_midpoint": "60.0",
            "judged_count": "2",
            "correct_count": "1",
            "accuracy": "0.5",
            "ci_lower": "0.09452865480086614",
            "ci_upper": "0.9054713451991339",
        }
    ]


def test_kv_precompute_copies_its_time_and_every_precompute_column():
    row = {"memory_setup_time_ms": 5.0, "kv_precomputed_chunks": "stale"}
    kv_metrics = {
        "kv_precompute_time_ms": "1234.5",
        "kv_precomputed_chunks": 10,
        "kv_precomputed_gpu_mb": 1.5,
        "kv_chunk_cache_residency_is_gpu": True,
        "llama_kv_chunk_count": 10,
        "llama_kv_total_tensor_gpu_mb": 3.25,
        "kv_precomputed_but_not_a_column": 1,
        "lmcache_warm_prompts": 3,
    }

    record_kv_precompute(row, kv_metrics)

    copied_columns = [
        column
        for column in SAMPLE_SETUP_COLUMNS
        if column.startswith(("kv_precomputed_", "kv_chunk_cache_", "llama_kv_"))
    ]
    assert list(row) == [
        "memory_setup_time_ms",
        "kv_precomputed_chunks",
        "kv_precompute_time_ms",
    ] + [column for column in copied_columns if column != "kv_precomputed_chunks"]
    assert len(copied_columns) == 17
    assert {column: cell for column, cell in row.items() if cell is not None} == {
        "memory_setup_time_ms": 5.0,
        "kv_precompute_time_ms": 1234.5,
        "kv_precomputed_chunks": 10.0,
        "kv_precomputed_gpu_mb": 1.5,
        "llama_kv_chunk_count": 10.0,
        "llama_kv_total_tensor_gpu_mb": 3.25,
    }


def test_memory_warmup_copies_the_lmcache_columns():
    row = {"lmcache_warm_prompts": "stale"}

    record_memory_warmup(
        row, {"lmcache_warm_time_ms": 50, "lmcache_warm_tokens": "3000", "kv_precomputed_chunks": 1}
    )

    assert row == {
        "lmcache_warm_prompts": None,
        "lmcache_warm_time_ms": 50.0,
        "lmcache_warm_tokens": 3000.0,
        "lmcache_cpu_cache_mb": None,
    }


@pytest.mark.parametrize(
    "row,total",
    [
        (
            {
                "memory_setup_time_ms": 100.5,
                "kv_precompute_time_ms": 4000,
                "answer_prepare_sample_time_ms": "3.5",
            },
            4104.0,
        ),
        (
            {
                "memory_create_time_ms": 1.0,
                "vector_index_build_time_ms": 2.0,
                "sample_setup_time_ms": 3.0,
                "answer_prepare_sample_time_ms": 4.0,
            },
            4.0,
        ),
    ],
)
def test_setup_total_adds_memory_setup_kv_precompute_and_sample_preparation(row, total):
    assert setup_total_ms(row) == total
    assert type(setup_total_ms(row)) is type(total)


def test_query_metrics_csv_columns_are_pinned():
    assert QUERY_METRICS_COLUMNS == [
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
    assert ACCURACY_BIN_COLUMNS == [
        "token_bin_start",
        "token_bin_end",
        "token_bin_midpoint",
        "judged_count",
        "correct_count",
        "accuracy",
        "ci_lower",
        "ci_upper",
    ]


def test_sample_setup_csv_columns_are_pinned():
    assert SAMPLE_SETUP_COLUMNS == [
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
        "lmcache_warm_time_ms",
        "lmcache_warm_prompts",
        "lmcache_warm_tokens",
        "lmcache_cpu_cache_mb",
        "answer_prepare_sample_time_ms",
        "sample_setup_time_ms",
    ]


def test_every_numeric_and_list_metric_column_is_read_from_the_metric_of_the_same_name():
    numbers = {column: float(position) for position, column in enumerate(NUMERIC_METRIC_COLUMNS)}
    lists = {column: [column] for column in LIST_METRIC_COLUMNS}

    row = query_row({**numbers, **lists})

    assert {column: row[column] for column in numbers} == numbers
    assert {column: row[column] for column in lists} == lists
    assert len(numbers) == 18
    assert list(lists) == [
        "memory_retrieved_fact_ids",
        "memory_retrieved_fact_text_hashes",
        "memory_injected_fact_ids",
        "memory_context_turn_ids",
    ]
