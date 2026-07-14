from __future__ import annotations

from .config import BenchmarkConfig


def result_mode(config: BenchmarkConfig) -> str:
    return _mem0_result_mode(config.answer_backend)


def existing_run_mode(
    saved_config: dict[str, object],
    records: list[dict[str, object]],
    fallback_config: BenchmarkConfig,
) -> str:
    answer_backend = saved_config.get("answer_backend")
    if (
        saved_config.get("memory_unit") == "mem0-fact"
        and answer_backend in {"vllm-kv", "vllm-prefix"}
    ):
        assert isinstance(answer_backend, str)
        return _mem0_result_mode(answer_backend)
    if isinstance(answer_backend, str) and answer_backend:
        return answer_backend
    if records and isinstance(records[0].get("mode"), str):
        return str(records[0]["mode"])
    return result_mode(fallback_config)


def _mem0_result_mode(answer_backend: str) -> str:
    if answer_backend == "vllm-kv":
        return "mem0-kv"
    if answer_backend == "vllm-prefix":
        return "mem0-prefix"
    raise ValueError(f"Unsupported answer backend: {answer_backend!r}.")
