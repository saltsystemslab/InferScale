from __future__ import annotations

import json
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable


class JsonlWriter(AbstractContextManager["JsonlWriter"]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_records(
    records: Iterable[dict[str, Any]],
    *,
    run_id: str,
    mode: str,
    config: dict[str, Any],
    system_metadata: dict[str, Any],
) -> dict[str, Any]:
    rows = list(records)
    judged = [row for row in rows if row.get("judge", {}).get("correct") is not None]
    correct = sum(1 for row in judged if row.get("judge", {}).get("correct") is True)
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "judged": 0, "correct": 0})
    for row in rows:
        category = str(row.get("category") or "unknown")
        by_category[category]["count"] += 1
        judged_value = row.get("judge", {}).get("correct")
        if judged_value is not None:
            by_category[category]["judged"] += 1
        if judged_value is True:
            by_category[category]["correct"] += 1
    for stats in by_category.values():
        stats["accuracy"] = _safe_div(stats["correct"], stats["judged"])

    latency_keys = [
        "memory_search_ms",
        "answer_generation_ms",
        "judge_ms",
        "end_to_end_ms",
    ]
    latencies = {
        key: _average(row.get("latency_ms", {}).get(key) for row in rows)
        for key in latency_keys
    }

    jasper = _summarize_jasper(rows)
    return {
        "run_id": run_id,
        "mode": mode,
        "question_count": len(rows),
        "judged_count": len(judged),
        "correct_count": correct,
        "accuracy": _safe_div(correct, len(judged)),
        "by_category": dict(sorted(by_category.items())),
        "latency_avg_ms": latencies,
        "jasper": jasper,
        "config": config,
        "system": system_metadata,
    }


def _summarize_jasper(rows: list[dict[str, Any]]) -> dict[str, Any]:
    build_metrics = [row.get("index", {}) for row in rows]
    search_metrics = [row.get("memory", {}) for row in rows]
    vector_counts = [item.get("indexed_vector_count") for item in build_metrics if item.get("indexed_vector_count") is not None]
    dims = [item.get("embedding_dim") for item in build_metrics if item.get("embedding_dim") is not None]
    return {
        "backend": next((item.get("backend") for item in build_metrics if item.get("backend")), None),
        "graph_build_time_ms_max": max((item.get("graph_build_time_ms", 0.0) for item in build_metrics), default=0.0),
        "search_time_ms_avg": _average(item.get("vector_search_ms") for item in search_metrics),
        "indexed_vector_count_max": max(vector_counts, default=0),
        "embedding_dim": dims[0] if dims else None,
    }


def _average(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
