from __future__ import annotations

import csv
from pathlib import Path

import pytest

from memory_config import make_memory_config

from benchmarks.common.clients import ChatResult
from benchmarks.common.vector_types import RetrievalMetrics
from benchmarks.memory.data import ConversationSample, QuestionAnswer
from benchmarks.memory.evaluation import QuestionEvaluator
from benchmarks.memory.mem0.profiling import RETRIEVAL_STAGE_METRIC_KEYS
from benchmarks.memory.reporting import QUERY_METRICS_COLUMNS, query_metric_rows
from benchmarks.common.files import write_csv
from benchmarks.memory.results import summarize_records
from benchmarks.memory.runtime_clients import RuntimeClients


def _record(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "run_id": "run",
        "mode": "full",
        "sample_id": "sample",
        "question_id": "question",
        "category": "single-hop",
        "retrieved_memories": [{"turn_id": "turn-2"}, {"turn_id": "turn-4"}],
        "metrics": metrics,
        "judge": {"correct": True},
    }


def _judged_record(category: str, correct: bool | None) -> dict[str, object]:
    record = _record({})
    record["category"] = category
    record["judge"] = {"correct": correct}
    return record


def test_summary_reports_accuracy_by_upstream_category() -> None:
    rows = [
        _judged_record("1", True),
        _judged_record("1", False),
        _judged_record("2", True),
        _judged_record("4", None),
    ]

    summary = summarize_records(
        rows,
        run_id="run",
        mode="full",
        config={},
        system_metadata={},
    )
    by_category = summary["metrics"]["accuracy_by_category"]

    assert by_category["1"] == {
        "name": "multi-hop",
        "total": 2,
        "correct": 1,
        "accuracy": 0.5,
    }
    assert by_category["2"] == {
        "name": "temporal",
        "total": 1,
        "correct": 1,
        "accuracy": 1.0,
    }
    assert "4" not in by_category  # unjudged records are excluded


def test_summary_aggregates_kv_verify_time() -> None:
    rows = [
        _record({"kv_verify_time_ms": 10.0}),
        _record({"kv_verify_time_ms": 30.0}),
    ]

    summary = summarize_records(
        rows,
        run_id="run",
        mode="full",
        config={},
        system_metadata={},
    )

    assert summary["metrics"]["kv_verify_time_ms"]["avg"] == 20.0


def test_summary_aggregates_prompt_injection_timing() -> None:
    key = "prompt_injection_engine_time_to_first_token_ms"
    rows = [
        _record({key: 10.0}),
        _record({key: 20.0}),
        _record({key: 30.0}),
    ]

    metrics = summarize_records(
        rows, run_id="run", mode="mem0-prompt-injection", config={}, system_metadata={},
    )["metrics"]

    assert metrics[key]["count"] == 3
    assert metrics[key]["avg"] == 20.0


def test_summary_aggregates_only_profiled_deepcopies_without_adjusting_latency() -> None:
    rows = [
        _record({"vector_db_query_time_ms": 10.0, "query_to_answer_ms": 100.0}),
        _record({
            "vector_db_query_time_ms": 20.0, "query_to_answer_ms": 200.0,
            "qdrant_deepcopy_time_ms": 0.0, "qdrant_deepcopy_calls": 0,
            "qdrant_entity_deepcopy_time_ms": 0.0,
            "qdrant_entity_deepcopy_wall_time_ms": 0.0,
            "qdrant_entity_deepcopy_calls": 0,
        }),
        _record({
            "vector_db_query_time_ms": 30.0, "query_to_answer_ms": 300.0,
            "qdrant_deepcopy_time_ms": 4.0, "qdrant_deepcopy_calls": 6,
            "qdrant_entity_deepcopy_time_ms": 8.0,
            "qdrant_entity_deepcopy_wall_time_ms": 5.0,
            "qdrant_entity_deepcopy_calls": 12,
        }),
    ]
    metrics = summarize_records(
        rows, run_id="run", mode="full", config={}, system_metadata={},
    )["metrics"]

    assert metrics["qdrant_deepcopy_time_ms"]["count"] == 2
    assert metrics["qdrant_deepcopy_time_ms"]["avg"] == 2.0
    assert metrics["qdrant_deepcopy_calls"]["count"] == 2
    assert metrics["qdrant_deepcopy_calls"]["avg"] == 3.0
    assert metrics["qdrant_entity_deepcopy_time_ms"]["count"] == 2
    assert metrics["qdrant_entity_deepcopy_time_ms"]["avg"] == 4.0
    assert metrics["qdrant_entity_deepcopy_wall_time_ms"]["count"] == 2
    assert metrics["qdrant_entity_deepcopy_wall_time_ms"]["avg"] == 2.5
    assert metrics["qdrant_entity_deepcopy_calls"]["count"] == 2
    assert metrics["qdrant_entity_deepcopy_calls"]["avg"] == 6.0
    assert metrics["vector_db_query_time_total_ms"] == 60.0
    assert metrics["vector_db_query_time_ms"]["avg"] == 20.0
    assert metrics["query_to_answer_ms"]["avg"] == 200.0


def test_unprofiled_summary_omits_deepcopy_diagnostics() -> None:
    diagnostic_keys = (
        "qdrant_deepcopy_time_ms", "qdrant_deepcopy_calls",
        "qdrant_entity_deepcopy_time_ms", "qdrant_entity_deepcopy_wall_time_ms",
        "qdrant_entity_deepcopy_calls",
    )
    metrics = summarize_records(
        [_record(dict.fromkeys(diagnostic_keys))],
        run_id="run", mode="full", config={}, system_metadata={},
    )["metrics"]
    assert not set(diagnostic_keys) & metrics.keys()


@pytest.mark.parametrize(
    "copy_ms,copy_calls,entity_ms,entity_wall_ms,entity_calls",
    [(None, None, None, None, None), (0.0, 0, 0.0, 0.0, 0), (2.5, 3, 8.0, 5.5, 7)],
)
def test_query_metrics_csv_preserves_optional_deepcopy_diagnostics(
    tmp_path: Path, copy_ms: float | None, copy_calls: int | None,
    entity_ms: float | None, entity_wall_ms: float | None, entity_calls: int | None,
) -> None:
    diagnostics = {
        "qdrant_deepcopy_time_ms": copy_ms,
        "qdrant_deepcopy_calls": copy_calls,
        "qdrant_entity_deepcopy_time_ms": entity_ms,
        "qdrant_entity_deepcopy_wall_time_ms": entity_wall_ms,
        "qdrant_entity_deepcopy_calls": entity_calls,
    }
    row = query_metric_rows([_record({
        **diagnostics,
        "query_to_answer_ms": 100.0,
    })])[0]
    assert {key: row[key] for key in diagnostics} == diagnostics
    assert row["query_to_answer_ms"] == 100.0

    path = tmp_path / "query_metrics.csv"
    write_csv(path, [row], QUERY_METRICS_COLUMNS)
    with path.open(newline="", encoding="utf-8") as fh:
        written = next(csv.DictReader(fh))
    for key, value in diagnostics.items():
        assert written[key] == ("" if value is None else str(value))


@pytest.mark.parametrize("stage_ms", [None, 0.0, 2.5])
def test_retrieval_stage_metrics_survive_record_summary_and_csv(
    tmp_path: Path, stage_ms: float | None,
) -> None:
    config = make_memory_config(run_id="stages", skip_judge=True)
    qa = QuestionAnswer("sample", "question", "What?", "Answer", "1")
    sample = ConversationSample("sample", [], [qa], {})
    diagnostics = (
        None if stage_ms is None else dict.fromkeys(RETRIEVAL_STAGE_METRIC_KEYS, stage_ms)
    )
    if diagnostics is not None:
        # Diagnostic extensions must not overwrite the headline timing fields.
        diagnostics.update({
            "query_to_first_token_ms": 999.0,
            "query_retrieval_time_ms": 999.0,
            "unknown_diagnostic": 999.0,
        })
    evaluator = QuestionEvaluator(config, RuntimeClients(answer_client=None, judge_client=None))
    record = evaluator.record_answer(
        sample, qa, [],
        ChatResult(content="Answer", ttft_ms=3.0, metrics={"query_to_first_token_ms": 14.0}),
        retrieval_metrics=RetrievalMetrics(1.0, 2.0, 5.0, stage_timings=diagnostics),
    )
    summary = summarize_records(
        [record], run_id="stages", mode=record["mode"], config={}, system_metadata={},
    )["metrics"]
    row = query_metric_rows([record])[0]
    path = tmp_path / "query_metrics.csv"
    write_csv(path, [row], QUERY_METRICS_COLUMNS)
    with path.open(newline="", encoding="utf-8") as fh:
        written = next(csv.DictReader(fh))

    assert record["metrics"]["query_retrieval_time_ms"] == 5.0
    assert record["metrics"]["query_to_first_token_ms"] == 14.0
    assert "unknown_diagnostic" not in record["metrics"]
    assert "stage_timings" not in record["metrics"]
    assert summary["query_retrieval_time_ms"]["avg"] == 5.0
    assert summary["query_to_first_token_ms"]["avg"] == 14.0
    assert written["query_to_first_token_ms"] == "14.0"
    for key in RETRIEVAL_STAGE_METRIC_KEYS:
        if stage_ms is None:
            assert key not in record["metrics"]
            assert key not in summary
            assert row[key] is None
            assert written[key] == ""
        else:
            assert record["metrics"][key] == stage_ms
            assert summary[key]["count"] == 1
            assert summary[key]["avg"] == stage_ms
            assert row[key] == stage_ms
            assert written[key] == str(stage_ms)


def test_retrieval_stage_summary_counts_only_profiled_queries() -> None:
    rows = [
        _record({}),
        _record(dict.fromkeys(RETRIEVAL_STAGE_METRIC_KEYS)),
        _record(dict.fromkeys(RETRIEVAL_STAGE_METRIC_KEYS, 0.0)),
        _record(dict.fromkeys(RETRIEVAL_STAGE_METRIC_KEYS, 4.0)),
    ]
    metrics = summarize_records(
        rows, run_id="stages", mode="mem0-prompt-injection", config={}, system_metadata={},
    )["metrics"]
    for key in RETRIEVAL_STAGE_METRIC_KEYS:
        assert metrics[key]["count"] == 2
        assert metrics[key]["avg"] == 2.0


@pytest.mark.parametrize(
    "query_ms,query_calls,entity_ms,entity_wall_ms,entity_calls",
    [(None, None, None, None, None), (0.0, 0, 0.0, 0.0, 0), (0.5, 1, 2.0, 1.25, 3)],
)
def test_query_deepcopy_metrics_survive_record_summary_and_csv(
    tmp_path: Path, query_ms: float | None, query_calls: int | None,
    entity_ms: float | None, entity_wall_ms: float | None, entity_calls: int | None,
) -> None:
    diagnostics = {
        "qdrant_query_deepcopy_time_ms": query_ms,
        "qdrant_query_deepcopy_calls": query_calls,
        "qdrant_entity_query_deepcopy_time_ms": entity_ms,
        "qdrant_entity_query_deepcopy_wall_time_ms": entity_wall_ms,
        "qdrant_entity_query_deepcopy_calls": entity_calls,
    }
    config = make_memory_config(run_id="copies", skip_judge=True)
    qa = QuestionAnswer("sample", "question", "What?", "Answer", "1")
    sample = ConversationSample("sample", [], [qa], {})
    evaluator = QuestionEvaluator(config, RuntimeClients(answer_client=None, judge_client=None))
    record = evaluator.record_answer(
        sample, qa, [],
        ChatResult(content="Answer", ttft_ms=3.0, metrics={"query_to_first_token_ms": 14.0}),
        retrieval_metrics=RetrievalMetrics(
            1.0, 2.0, 5.0, qdrant_deepcopy_time_ms=0.75,
            qdrant_entity_deepcopy_time_ms=4.0, **diagnostics,
        ),
    )
    summary = summarize_records(
        [record, _record({})], run_id="copies", mode=record["mode"], config={}, system_metadata={},
    )["metrics"]
    row = query_metric_rows([record])[0]
    path = tmp_path / "query_metrics.csv"
    write_csv(path, [row], QUERY_METRICS_COLUMNS)
    with path.open(newline="", encoding="utf-8") as fh:
        written = next(csv.DictReader(fh))

    assert QUERY_METRICS_COLUMNS[-5:] == list(diagnostics)
    assert record["metrics"]["qdrant_deepcopy_time_ms"] == 0.75
    assert record["metrics"]["qdrant_entity_deepcopy_time_ms"] == 4.0
    assert summary["query_to_first_token_ms"]["avg"] == 14.0
    assert summary["vector_db_query_time_ms"]["avg"] == 2.0
    assert summary["query_retrieval_time_ms"]["avg"] == 5.0
    for key, value in diagnostics.items():
        assert record["metrics"][key] == value
        assert row[key] == value
        assert written[key] == ("" if value is None else str(value))
        if value is None:
            assert key not in summary
        else:
            assert summary[key]["count"] == 1
            assert summary[key]["avg"] == value


def test_query_metrics_expose_backend_neutral_memory_audit_fields(tmp_path: Path) -> None:
    record = _record(
        {
            "kv_memory_tokens": 40,
            "kv_query_tokens": 10,
            "memory_context_window": 2,
            "memory_retrieved_fact_ids": ["fact-2", "fact-4"],
            "memory_retrieved_fact_text_hashes": ["sha256:two", "sha256:four"],
            "memory_injected_fact_ids": ["fact-4", "fact-2"],
            "memory_context_turn_ids": ["turn-1", "turn-3"],
            "memory_context_turn_count": 2,
            "memory_context_encoding_tokens_total": 81,
            "memory_context_encoding_tokens_max": 43,
            "memory_context_encoding_truncated_tokens": 0,
            "memory_token_budget": 64,
            "kv_block_size": 16,
            "kv_loaded_memory_tokens": 32,
            "kv_recomputed_memory_tail_tokens": 8,
            "kv_fact_tokens_end": 30,
            "kv_verify_time_ms": 12.5,
        }
    )

    row = query_metric_rows([record])[0]

    assert row["memory_tokens"] == 40
    assert row["query_tokens"] == 10
    assert row["total_prompt_tokens"] == 50
    assert row["memory_context_window"] == 2
    assert row["memory_retrieved_fact_ids"] == ["fact-2", "fact-4"]
    assert row["memory_retrieved_fact_text_hashes"] == ["sha256:two", "sha256:four"]
    assert row["memory_injected_fact_ids"] == ["fact-4", "fact-2"]
    assert row["memory_context_turn_ids"] == ["turn-1", "turn-3"]
    assert row["memory_context_turn_count"] == 2
    assert row["memory_context_encoding_tokens_total"] == 81
    assert row["memory_context_encoding_tokens_max"] == 43
    assert row["memory_context_encoding_truncated_tokens"] == 0
    assert row["memory_token_budget"] == 64
    assert row["kv_block_size"] == 16
    assert row["kv_loaded_memory_tokens"] == 32
    assert row["kv_recomputed_memory_tail_tokens"] == 8
    assert row["kv_fact_tokens_end"] == 30
    assert row["kv_verify_time_ms"] == 12.5

    path = tmp_path / "query_metrics.csv"
    write_csv(path, [row], QUERY_METRICS_COLUMNS)
    with path.open(newline="", encoding="utf-8") as fh:
        written = next(csv.DictReader(fh))
    assert written["memory_injected_fact_ids"] == '["fact-4","fact-2"]'


def test_historical_kv_metrics_remain_readable() -> None:
    record = _record(
        {
            "kv_memory_tokens": 31,
            "kv_query_tokens": 11,
            "kv_context_window": 3,
        }
    )

    row = query_metric_rows([record])[0]
    summary = summarize_records(
        [record],
        run_id="run",
        mode="full",
        config={},
        system_metadata={},
    )

    assert row["memory_tokens"] == 31
    assert row["query_tokens"] == 11
    assert row["memory_context_window"] == 3
    assert summary["metrics"]["kv_context_window"]["avg"] == 3


def test_summary_aggregates_backend_neutral_memory_audit_metrics() -> None:
    rows = [
        _record(
            {
                "memory_context_window": 2,
                "memory_context_turn_count": 2,
                "memory_context_encoding_tokens_total": 41,
                "memory_context_encoding_tokens_max": 23,
                "memory_context_encoding_truncated_tokens": 0,
                "memory_token_budget": 64,
            }
        ),
        _record(
            {
                "memory_context_window": 2,
                "memory_context_turn_count": 2,
                "memory_context_encoding_tokens_total": 55,
                "memory_context_encoding_tokens_max": 29,
                "memory_context_encoding_truncated_tokens": 4,
                "memory_token_budget": 64,
            }
        ),
    ]

    summary = summarize_records(
        rows,
        run_id="run",
        mode="full",
        config={},
        system_metadata={},
    )
    metrics = summary["metrics"]

    assert metrics["memory_context_window"]["avg"] == 2
    assert metrics["memory_context_turn_count"]["avg"] == 2
    assert metrics["memory_context_encoding_tokens_total"]["avg"] == 48
    assert metrics["memory_context_encoding_tokens_max"]["max"] == 29
    assert metrics["memory_context_encoding_truncated_tokens"]["avg"] == 2
    assert metrics["memory_token_budget"]["avg"] == 64
