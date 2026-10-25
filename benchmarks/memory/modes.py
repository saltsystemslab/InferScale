"""The result mode recorded on every prediction, named after the answer backend."""

from __future__ import annotations

from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.protocol import LMCACHE_BACKEND


def result_mode(config: MemoryRunConfig) -> str:
    if config.answer_backend == "kv-injection":
        return "mem0-kv"
    if config.answer_backend == "prompt-injection":
        return "mem0-prompt-injection"
    if config.answer_backend == LMCACHE_BACKEND:
        return "mem0-lmcache-blend"
    raise ValueError(f"Unsupported answer backend: {config.answer_backend!r}.")
