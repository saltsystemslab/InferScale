from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.common.deferred_judging import judge_pending_records


def _records() -> list[dict[str, Any]]:
    return [
        {"question_id": "q0", "judge": {"correct": True}},
        {"question_id": "q1", "judge": {"correct": None}},
        {"question_id": "q2"},
    ]


def _verdict(record: dict[str, Any]) -> dict[str, Any]:
    return {"correct": record["question_id"] == "q2"}


def test_only_unjudged_records_are_judged_and_progress_is_saved_each_time(tmp_path: Path) -> None:
    records = _records()
    saves: list[tuple[int, bool]] = []

    def save(rows: list[dict[str, Any]], final: bool) -> dict[str, Any]:
        saves.append((sum(1 for row in rows if "judge" in row), final))
        return {"judged_count": len(saves)}

    summary, judged_now = judge_pending_records(
        records,
        predictions_path=tmp_path / "predictions.jsonl",
        rejudge=False,
        judge=_verdict,
        failed_payload=lambda exc: {"correct": None, "status": "error"},
        save=save,
        label=lambda record: record["question_id"],
    )

    assert judged_now == 2
    assert [record["judge"]["correct"] for record in records] == [True, False, True]
    assert saves == [(2, False), (3, False), (3, True)]
    assert summary == {"judged_count": 3}


def test_rejudge_judges_every_record(tmp_path: Path) -> None:
    records = _records()

    _, judged_now = judge_pending_records(
        records,
        predictions_path=tmp_path / "predictions.jsonl",
        rejudge=True,
        judge=_verdict,
        failed_payload=lambda exc: {},
        save=lambda rows, final: {},
        label=lambda record: record["question_id"],
    )

    assert judged_now == 3
    assert records[0]["judge"] == {"correct": False}


def test_failed_judge_request_is_recorded_saved_and_raised(tmp_path: Path) -> None:
    records = _records()
    saves: list[bool] = []

    def judge(record: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("judge server down")

    with pytest.raises(RuntimeError, match="row 2/3 q1.*judge server down"):
        judge_pending_records(
            records,
            predictions_path=tmp_path / "predictions.jsonl",
            rejudge=False,
            judge=judge,
            failed_payload=lambda exc: {"correct": None, "reason": str(exc)},
            save=lambda rows, final: saves.append(final) or {},
            label=lambda record: record["question_id"],
        )

    assert records[1]["judge"] == {"correct": None, "reason": "judge server down"}
    assert "judge" not in records[2]
    assert saves == [False]
