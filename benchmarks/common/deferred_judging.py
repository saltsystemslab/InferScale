"""Judge the unjudged records of a saved run, saving progress after every verdict."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.common.judge import is_judged

Record = dict[str, Any]
JudgeRecord = Callable[[Record], dict[str, Any]]
FailedPayload = Callable[[Exception], dict[str, Any]]
SaveRecords = Callable[[list[Record], bool], dict[str, Any]]
RecordLabel = Callable[[Record], str]


def judge_pending_records(
    records: list[Record],
    *,
    predictions_path: Path,
    rejudge: bool,
    judge: JudgeRecord,
    failed_payload: FailedPayload,
    save: SaveRecords,
    label: RecordLabel,
) -> tuple[dict[str, Any], int]:
    """Judge every unjudged record (every record when ``rejudge``) in place.

    ``save(records, final)`` persists the records and returns the run summary.
    It runs after every verdict, so an interrupted judge stage resumes where it
    stopped, and once more with ``final=True`` when every record is judged.
    A failed judge request is recorded on its record, saved, and re-raised.
    Returns the final summary and the number of records judged by this call.
    """
    judged_now = 0
    for row_number, record in enumerate(records, start=1):
        if not rejudge and is_judged(record):
            continue
        try:
            record["judge"] = judge(record)
        except Exception as exc:
            record["judge"] = failed_payload(exc)
            save(records, False)
            raise RuntimeError(
                f"Judge request failed for row {row_number}/{len(records)} {label(record)}. "
                f"Saved progress to {predictions_path}; fix or restart the judge server, "
                f"then rerun the judge stage. Original error: {type(exc).__name__}: {exc}"
            ) from exc
        judged_now += 1
        save(records, False)
    return save(records, True), judged_now
