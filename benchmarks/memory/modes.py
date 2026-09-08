from __future__ import annotations

from benchmarks.memory.config import MemoryRunConfig


def result_mode(config: MemoryRunConfig) -> str:
    return _mem0_result_mode(config.answer_backend)


def existing_run_mode(
    saved_config: dict[str, object],
    records: list[dict[str, object]],
    fallback_config: MemoryRunConfig,
) -> str:
    answer_backend = saved_config.get("answer_backend")
    if saved_config.get("memory_unit") == "mem0-fact" and isinstance(answer_backend, str):
        # Historical result files retain the names used when they were written.
        current_backend = {
            "vllm-kv": "kv-injection",
            "vllm-prefix": "prompt-injection",
        }.get(answer_backend, answer_backend)
        if current_backend in {"kv-injection", "prompt-injection"}:
            return _mem0_result_mode(current_backend)
    if isinstance(answer_backend, str) and answer_backend:
        return answer_backend
    if records and isinstance(records[0].get("mode"), str):
        return str(records[0]["mode"])
    return result_mode(fallback_config)


def _mem0_result_mode(answer_backend: str) -> str:
    if answer_backend == "kv-injection":
        return "mem0-kv"
    if answer_backend == "prompt-injection":
        return "mem0-prefix"
    raise ValueError(f"Unsupported answer backend: {answer_backend!r}.")
