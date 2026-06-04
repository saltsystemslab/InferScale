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

    vector_store = _summarize_vector_store(rows)
    return {
        "run_id": run_id,
        "mode": mode,
        "question_count": len(rows),
        "judged_count": len(judged),
        "correct_count": correct,
        "accuracy": _safe_div(correct, len(judged)),
        "by_category": dict(sorted(by_category.items())),
        "latency_avg_ms": latencies,
        "latency_ms": {
            key: _numeric_summary(row.get("latency_ms", {}).get(key) for row in rows)
            for key in latency_keys
        },
        "vllm": {
            "answer": _summarize_vllm(rows, "answer"),
            "judge": _summarize_vllm(rows, "judge"),
        },
        "retrieval": _summarize_retrieval(rows),
        "vector_store": vector_store,
        "jasper": vector_store,
        "config": config,
        "system": system_metadata,
    }


def _summarize_vector_store(rows: list[dict[str, Any]]) -> dict[str, Any]:
    build_metrics = [row.get("index", {}) for row in rows]
    search_metrics = [row.get("memory", {}) for row in rows]
    vector_counts = [item.get("indexed_vector_count") for item in build_metrics if item.get("indexed_vector_count") is not None]
    dims = [item.get("embedding_dim") for item in build_metrics if item.get("embedding_dim") is not None]
    return {
        "backend": next((item.get("backend") for item in build_metrics if item.get("backend")), None),
        "graph_build_time_ms_max": max((item.get("graph_build_time_ms", 0.0) for item in build_metrics), default=0.0),
        "search_time_ms_avg": _average(item.get("vector_search_ms") for item in search_metrics),
        "search_time_ms": _numeric_summary(item.get("vector_search_ms") for item in search_metrics),
        "indexed_vector_count_max": max(vector_counts, default=0),
        "embedding_dim": dims[0] if dims else None,
    }


def _summarize_vllm(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    metrics = [row.get("vllm", {}).get(role, {}) or {} for row in rows]
    return {
        "latency_ms": _numeric_summary(item.get("latency_ms") for item in metrics),
        "ttft_ms": _numeric_summary(item.get("ttft_ms") for item in metrics),
        "output_tokens_per_sec": _numeric_summary(item.get("output_tokens_per_sec") for item in metrics),
        "prompt_tokens": _numeric_summary(item.get("prompt_tokens") for item in metrics),
        "completion_tokens": _numeric_summary(item.get("completion_tokens") for item in metrics),
        "total_tokens": _numeric_summary(item.get("total_tokens") for item in metrics),
    }


def _summarize_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrieved_counts = [len(row.get("retrieved_memories") or []) for row in rows]
    duplicate_id_questions = 0
    duplicate_turn_questions = 0
    duplicate_id_count = 0
    duplicate_turn_id_count = 0
    for row in rows:
        memories = row.get("retrieved_memories") or []
        ids = [str(item.get("id")) for item in memories if item.get("id") is not None]
        turn_ids = []
        for item in memories:
            metadata = item.get("metadata")
            if isinstance(metadata, dict) and metadata.get("turn_id") is not None:
                turn_ids.append(str(metadata["turn_id"]))
        id_duplicates = len(ids) - len(set(ids))
        turn_duplicates = len(turn_ids) - len(set(turn_ids))
        if id_duplicates:
            duplicate_id_questions += 1
            duplicate_id_count += id_duplicates
        if turn_duplicates:
            duplicate_turn_questions += 1
            duplicate_turn_id_count += turn_duplicates
    return {
        "retrieved_count": _numeric_summary(retrieved_counts),
        "questions_with_duplicate_ids": duplicate_id_questions,
        "questions_with_duplicate_turn_ids": duplicate_turn_questions,
        "duplicate_id_count": duplicate_id_count,
        "duplicate_turn_id_count": duplicate_turn_id_count,
    }


def _average(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    numbers = sorted(float(value) for value in values if value is not None)
    if not numbers:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(numbers),
        "avg": sum(numbers) / len(numbers),
        "min": numbers[0],
        "p50": _percentile(numbers, 0.50),
        "p95": _percentile(numbers, 0.95),
        "max": numbers[-1],
    }


def _percentile(sorted_numbers: list[float], percentile: float) -> float:
    if len(sorted_numbers) == 1:
        return sorted_numbers[0]
    index = (len(sorted_numbers) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(sorted_numbers) - 1)
    weight = index - lower
    return sorted_numbers[lower] * (1 - weight) + sorted_numbers[upper] * weight


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
