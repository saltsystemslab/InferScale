"""Numeric summaries shared by the result writers."""

from __future__ import annotations

from typing import Any, Iterable


def percentile(sorted_numbers: list[float], fraction: float) -> float:
    if len(sorted_numbers) == 1:
        return sorted_numbers[0]
    index = (len(sorted_numbers) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(sorted_numbers) - 1)
    weight = index - lower
    return sorted_numbers[lower] * (1 - weight) + sorted_numbers[upper] * weight


def number_values(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    numbers = sorted(number_values(values))
    if not numbers:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(numbers),
        "avg": sum(numbers) / len(numbers),
        "min": numbers[0],
        "p50": percentile(numbers, 0.50),
        "p95": percentile(numbers, 0.95),
        "max": numbers[-1],
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def queries_per_second(query_count: int, total_ms: float) -> float | None:
    if query_count == 0 or total_ms <= 0:
        return None
    return query_count / (total_ms / 1000)
