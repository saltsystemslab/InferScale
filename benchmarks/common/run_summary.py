from __future__ import annotations

from typing import Any

from benchmarks.common.judge import is_judged

Record = dict[str, Any]
REPORTED_LATENCIES = ("time_to_first_token_ms", "query_to_first_token_ms")


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


def median_latency_text(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    return " ".join(f"{name}={_median_text(metrics.get(name))}" for name in REPORTED_LATENCIES)


def _median_text(latency: Any) -> str:
    median = latency.get("p50") if isinstance(latency, dict) else None
    return "n/a" if median is None else f"{median:.1f}"
