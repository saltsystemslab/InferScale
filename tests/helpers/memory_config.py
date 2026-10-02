"""Build memory test configurations through the current JSON configuration API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.common.config import RuntimeConfig, jsonable_value, load_json_object
from benchmarks.memory.config import MemoryRunConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def memory_config_data(**values: Any) -> dict[str, Any]:
    data = load_json_object(PROJECT_ROOT / "configs/memory/accuracy-latency/llama.json")
    data.update(jsonable_value(values))
    return data


def memory_runtime(**values: Any) -> RuntimeConfig:
    data = load_json_object(PROJECT_ROOT / "configs/runtime.json")
    data["storage"] = {}
    data.update(jsonable_value(values))
    return RuntimeConfig.from_dict(data, root=PROJECT_ROOT)


def make_memory_config(
    *,
    runtime: RuntimeConfig | None = None,
    **values: Any,
) -> MemoryRunConfig:
    return MemoryRunConfig.from_dict(
        memory_config_data(**values),
        runtime=runtime if runtime is not None else memory_runtime(),
    )
