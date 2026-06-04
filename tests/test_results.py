from locomo_jasper_bench.results import summarize_records


def test_summarize_records_accuracy_by_category():
    summary = summarize_records(
        [
            {
                "category": "single-hop",
                "judge": {"correct": True},
                "latency_ms": {"memory_search_ms": 1, "answer_generation_ms": 2, "judge_ms": 3, "end_to_end_ms": 4},
                "memory": {"vector_search_ms": 0.5},
                "vllm": {
                    "answer": {
                        "latency_ms": 2,
                        "ttft_ms": 0.5,
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                        "output_tokens_per_sec": 20,
                    }
                },
                "retrieved_memories": [
                    {"id": "a", "metadata": {"turn_id": "t1"}},
                    {"id": "b", "metadata": {"turn_id": "t2"}},
                ],
                "index": {"backend": "numpy", "indexed_vector_count": 2, "embedding_dim": 3},
            },
            {
                "category": "single-hop",
                "judge": {"correct": False},
                "latency_ms": {"memory_search_ms": 3, "answer_generation_ms": 4, "judge_ms": 5, "end_to_end_ms": 6},
                "memory": {"vector_search_ms": 1.5},
                "vllm": {
                    "answer": {
                        "latency_ms": 4,
                        "ttft_ms": 1.5,
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                        "output_tokens_per_sec": 30,
                    }
                },
                "retrieved_memories": [
                    {"id": "c", "metadata": {"turn_id": "t3"}},
                    {"id": "c", "metadata": {"turn_id": "t3"}},
                ],
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
    assert summary["latency_ms"]["memory_search_ms"]["p50"] == 2.0
    assert summary["vllm"]["answer"]["ttft_ms"]["avg"] == 1.0
    assert summary["vllm"]["answer"]["output_tokens_per_sec"]["max"] == 30.0
    assert summary["vector_store"]["backend"] == "numpy"
    assert summary["vector_store"]["search_time_ms"]["avg"] == 1.0
    assert summary["jasper"]["search_time_ms_avg"] == 1.0
    assert summary["retrieval"]["questions_with_duplicate_ids"] == 1
    assert summary["retrieval"]["duplicate_turn_id_count"] == 1
