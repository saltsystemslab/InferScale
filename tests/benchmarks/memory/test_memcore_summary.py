from __future__ import annotations

from typing import Any

from benchmarks.memory.results import summarize_records

ALWAYS_REPORTED_METRICS = [
    "accuracy",
    "accuracy_by_category",
    "time_to_first_token_ms",
    "query_to_first_token_ms",
    "query_to_answer_ms",
    "query_embedding_time_ms",
    "query_retrieval_time_ms",
    "vector_db_query_time_ms",
    "vector_db_query_count",
    "vector_db_query_time_total_ms",
    "vector_db_queries_per_sec",
]
NO_NUMBERS = {"count": 0, "avg": None, "min": None, "p50": None, "p95": None, "max": None}


def record(metrics: Any = None, *, correct: Any = True, category: Any = "1") -> dict[str, Any]:
    return {
        "question_id": "q",
        "category": category,
        "judge": {"correct": correct},
        "metrics": {} if metrics is None else metrics,
    }


def summarize(records: Any, **options: Any) -> dict[str, Any]:
    return summarize_records(
        records,
        run_id="run",
        mode="mem0-kv",
        config={"saved": "config"},
        system_metadata={"saved": "system"},
        **options,
    )


def test_latencies_are_summarized_over_the_records_that_report_them():
    records = [
        record({"time_to_first_token_ms": 10.0, "query_to_answer_ms": 100}),
        record({"time_to_first_token_ms": 30.0, "query_to_answer_ms": None}),
        record({"time_to_first_token_ms": "20"}),
        record(metrics="not a dict"),
        {"question_id": "saved without metrics", "category": "1", "judge": {"correct": True}},
    ]

    metrics = summarize(records)["metrics"]

    assert metrics["time_to_first_token_ms"] == {
        "count": 3,
        "avg": 20.0,
        "min": 10.0,
        "p50": 20.0,
        "p95": 29.0,
        "max": 30.0,
    }
    assert metrics["query_to_answer_ms"]["count"] == 1
    assert metrics["query_to_first_token_ms"] == NO_NUMBERS


def test_setup_metrics_follow_the_query_metrics_and_leave_out_the_sample_sizes():
    setup_rows = [
        {
            "run_id": "run",
            "sample_id": "conv-1",
            "question_count": 100,
            "turn_count": 300,
            "sample_setup_time_ms": 90.0,
            "memory_setup_time_ms": 10.0,
            "kv_precompute_time_ms": None,
        },
        {"sample_id": "conv-2", "memory_setup_time_ms": 30.0, "not_a_setup_column": 1},
    ]

    metrics = summarize([record({"kv_block_size": 16})], sample_setup_metrics=iter(setup_rows))[
        "metrics"
    ]

    assert list(metrics) == [
        *ALWAYS_REPORTED_METRICS,
        "kv_block_size",
        "sample_setup_count",
        "memory_setup_time_ms",
        "sample_setup_time_ms",
    ]
    assert metrics["sample_setup_count"] == 2
    assert metrics["memory_setup_time_ms"]["avg"] == 20.0
    assert metrics["sample_setup_time_ms"]["count"] == 1


def test_accuracy_is_the_share_of_correct_answers_among_judged_records():
    records = [
        record(correct=True),
        record(correct=False),
        record(correct=True),
        record(correct=None),
        record(correct="true"),
    ]

    assert summarize(records)["metrics"]["accuracy"] == 2 / 3
    assert summarize([record(correct=None)])["metrics"]["accuracy"] is None


def test_saved_records_of_the_unscored_adversarial_category_are_still_summarized_by_name():
    by_category = summarize([record(category="5", correct=True)])["metrics"]["accuracy_by_category"]

    assert by_category == {"5": {"name": "adversarial", "total": 1, "correct": 1, "accuracy": 1.0}}


def test_optional_query_metrics_are_summarized_under_their_pinned_names_in_order():
    optional_metrics = [
        "memory_context_window",
        "memory_context_encoding_tokens_total",
        "memory_context_encoding_tokens_max",
        "memory_context_encoding_truncated_tokens",
        "memory_token_budget",
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
    ]
    reported = {
        name: float(position + 1) for position, name in enumerate(reversed(optional_metrics))
    }

    metrics = summarize([record(reported)])["metrics"]

    assert list(metrics) == [*ALWAYS_REPORTED_METRICS, *optional_metrics]
    assert {name: metrics[name]["avg"] for name in optional_metrics} == reported
