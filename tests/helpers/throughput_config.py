"""A ThroughputConfig for tests, with every storage path placed under a temporary directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.memory.throughput.config import ThroughputConfig


def make_throughput_config(tmp_path: Path, **overrides: Any) -> ThroughputConfig:
    settings: dict[str, Any] = {
        "model": "test/model",
        "model_label": "test",
        "results_dir": tmp_path,
        "run_id": "test-run",
        "cache_root": tmp_path / "cache",
        "embedding_cache_dir": tmp_path / "cache/embeddings",
        "memory_llm_cache_dir": tmp_path / "cache/mem0-inference",
        "local_store_dir": tmp_path / "scratch",
    }
    settings.update(overrides)
    return ThroughputConfig(**settings)
