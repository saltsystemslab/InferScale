from __future__ import annotations

import json

from locomo_jasper_bench.compare_runs import compare_run_summaries


def test_compare_run_summaries_reads_run_dirs_and_extracts_matrix_fields(tmp_path) -> None:
    run_dir = tmp_path / "jasper-bw128-norm"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "jasper-bw128-norm",
                "judged_count": 10,
                "question_count": 10,
                "metrics": {
                    "accuracy": 0.7,
                    "vector_db_query_time_ms": {"avg": 12.5},
                    "vector_db_queries_per_sec": 80.0,
                    "retrieval_diagnostics": {
                        "exact_recall_at_requested_top_k": {"avg": 0.95},
                    },
                },
                "config": {
                    "vector_backend": "jasper",
                    "judge_model": "google/gemma-3-12b-it",
                    "jasper_beam_width": 128,
                    "vector_normalize": True,
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_run_summaries([run_dir])

    assert comparison["run_count"] == 1
    assert comparison["runs"] == [
        {
            "run_id": "jasper-bw128-norm",
            "summary_path": str(run_dir / "summary.json"),
            "backend": "jasper",
            "judge_model": "google/gemma-3-12b-it",
            "jasper_beam_width": 128,
            "vector_normalize": True,
            "accuracy": 0.7,
            "judged_count": 10,
            "question_count": 10,
            "vector_db_query_time_ms_avg": 12.5,
            "vector_db_queries_per_sec": 80.0,
            "exact_recall_at_requested_top_k_avg": 0.95,
            "exact_top_k_answer_accuracy": None,
        }
    ]
