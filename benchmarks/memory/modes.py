from __future__ import annotations

from benchmarks.memory.config import MemoryRunConfig


def result_mode(config: MemoryRunConfig) -> str:
    if config.answer_backend == "kv-injection":
        return "mem0-kv"
    if config.answer_backend == "prompt-injection":
        return "mem0-prompt-injection"
    raise ValueError(f"Unsupported answer backend: {config.answer_backend!r}.")
