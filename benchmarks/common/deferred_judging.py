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

PROGRESS_SAVE = False
FINAL_SAVE = True


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
    judged_now = 0
    for row_number, record in enumerate(records, start=1):
        if not rejudge and is_judged(record):
            continue
        try:
            record["judge"] = judge(record)
        except Exception as exc:
            record["judge"] = failed_payload(exc)
            save(records, PROGRESS_SAVE)
            raise RuntimeError(
                f"Judge request failed for row {row_number}/{len(records)} {label(record)}. "
                f"Saved progress to {predictions_path}; fix or restart the judge server, "
                f"then rerun the judge stage. Original error: {type(exc).__name__}: {exc}"
            ) from exc
        judged_now += 1
        save(records, PROGRESS_SAVE)
    return save(records, FINAL_SAVE), judged_now
