"""The envelope of a run's summary.json, shared by the memory and RAG benchmarks."""

from __future__ import annotations

from typing import Any

from benchmarks.common.judge import is_judged

Record = dict[str, Any]


def judged_records(records: list[Record]) -> list[Record]:
    return [record for record in records if is_judged(record)]


def correct_count(judged: list[Record]) -> int:
    return sum(1 for record in judged if record["judge"]["correct"] is True)


def summary_envelope(
    records: list[Record],
    *,
    run_id: str,
    mode: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    system_metadata: dict[str, Any],
    **sections: Any,
) -> dict[str, Any]:
    """Counts, metrics, saved config, and host metadata; ``sections`` adds benchmark keys."""
    judged = judged_records(records)
    return {
        "run_id": run_id,
        "mode": mode,
        "question_count": len(records),
        "judged_count": len(judged),
        "correct_count": correct_count(judged),
        "metrics": metrics,
        **sections,
        "config": config,
        "system": system_metadata,
    }
