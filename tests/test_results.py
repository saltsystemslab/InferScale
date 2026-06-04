from locomo_jasper_bench.results import summarize_records


def test_summarize_records_reports_only_four_metrics():
    summary = summarize_records(
        [
            {
                "judge": {"correct": True},
                "metrics": {
                    "time_to_first_token_ms": 10,
                    "vector_db_query_time_ms": 2,
                    "throughput_tokens_per_sec": 40,
                },
            },
            {
                "judge": {"correct": False},
                "metrics": {
                    "time_to_first_token_ms": 20,
                    "vector_db_query_time_ms": 4,
                    "throughput_tokens_per_sec": 60,
                },
            },
        ],
        run_id="run",
        mode="baseline",
        config={},
        system_metadata={},
    )

    assert summary["question_count"] == 2
    assert summary["judged_count"] == 2
    assert summary["correct_count"] == 1
    assert set(summary) == {
        "run_id",
        "mode",
        "question_count",
        "judged_count",
        "correct_count",
        "metrics",
        "config",
        "system",
    }
    assert set(summary["metrics"]) == {
        "accuracy",
        "time_to_first_token_ms",
        "vector_db_query_time_ms",
        "throughput_tokens_per_sec",
    }
    assert summary["metrics"]["accuracy"] == 0.5
    assert summary["metrics"]["time_to_first_token_ms"]["avg"] == 15.0
    assert summary["metrics"]["vector_db_query_time_ms"]["p50"] == 3.0
    assert summary["metrics"]["throughput_tokens_per_sec"]["max"] == 60.0
