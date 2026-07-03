from __future__ import annotations

from .config import BenchmarkConfig


def result_mode(config: BenchmarkConfig) -> str:
    return config.answer_backend


def existing_run_mode(
    saved_config: dict[str, object],
    records: list[dict[str, object]],
    fallback_config: BenchmarkConfig,
) -> str:
    answer_backend = saved_config.get("answer_backend")
    if isinstance(answer_backend, str) and answer_backend:
        return answer_backend
    if records and isinstance(records[0].get("mode"), str):
        return str(records[0]["mode"])
    return result_mode(fallback_config)
