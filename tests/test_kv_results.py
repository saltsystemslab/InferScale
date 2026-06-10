from __future__ import annotations

from locomo_jasper_bench.results import summarize_records


def test_summarize_records_includes_kv_metrics() -> None:
    summary = summarize_records(
        [
            {
                "judge": {"correct": True},
                "metrics": {
                    "vector_db_query_time_ms": 2.0,
                    "kv_memory_tokens": 10,
                    "kv_compose_time_ms": 3.5,
                    "answer_generate_time_ms": 40.0,
                    "answer_total_time_ms": 45.0,
                    "kv_query_tokens": 8,
                    "kv_store_gpu_mb": 12.25,
                },
            },
            {
                "judge": {"correct": False},
                "metrics": {
                    "vector_db_query_time_ms": 4.0,
                    "kv_memory_tokens": 20,
                    "kv_compose_time_ms": 4.5,
                    "answer_generate_time_ms": 60.0,
                    "answer_total_time_ms": 66.0,
                    "kv_query_tokens": 12,
                    "kv_store_gpu_mb": 13.25,
                },
            },
        ],
        run_id="run",
        mode="vllm-kv",
        config={},
        system_metadata={},
    )

    assert summary["mode"] == "vllm-kv"
    assert summary["metrics"]["accuracy"] == 0.5
    assert summary["metrics"]["kv_memory_tokens"]["avg"] == 15.0
    assert summary["metrics"]["kv_compose_time_ms"]["avg"] == 4.0
    assert summary["metrics"]["answer_generate_time_ms"]["max"] == 60.0
    assert summary["metrics"]["answer_total_time_ms"]["p50"] == 55.5
    assert summary["metrics"]["kv_query_tokens"]["min"] == 8.0
    assert summary["metrics"]["kv_store_gpu_mb"]["p50"] == 12.75
