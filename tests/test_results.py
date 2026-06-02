from locomo_jasper_bench.results import summarize_records


def test_summarize_records_accuracy_by_category():
    summary = summarize_records(
        [
            {
                "category": "single-hop",
                "judge": {"correct": True},
                "latency_ms": {"memory_search_ms": 1, "answer_generation_ms": 2, "judge_ms": 3, "end_to_end_ms": 4},
                "memory": {"vector_search_ms": 0.5},
                "index": {"backend": "numpy", "indexed_vector_count": 2, "embedding_dim": 3},
            },
            {
                "category": "single-hop",
                "judge": {"correct": False},
                "latency_ms": {"memory_search_ms": 3, "answer_generation_ms": 4, "judge_ms": 5, "end_to_end_ms": 6},
                "memory": {"vector_search_ms": 1.5},
                "index": {"backend": "numpy", "indexed_vector_count": 2, "embedding_dim": 3},
            },
        ],
        run_id="run",
        mode="baseline",
        config={},
        system_metadata={},
    )

    assert summary["accuracy"] == 0.5
    assert summary["by_category"]["single-hop"]["accuracy"] == 0.5
    assert summary["latency_avg_ms"]["memory_search_ms"] == 2.0
    assert summary["jasper"]["search_time_ms_avg"] == 1.0
