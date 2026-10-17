"""The ``extraction`` section: how the extract-facts stage serves the answer model for Mem0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from benchmarks.common.json_fields import ConfigError, optional, reject_unknown_keys

EXTRACTION_KEYS = (
    "port",
    "workers",
    "gpu_memory_utilization",
    "health_timeout_s",
    "structured_outputs",
    "extra_vllm_args",
)


@dataclass(slots=True, frozen=True)
class ExtractionConfig:
    port: int = 8000
    workers: int = 4
    gpu_memory_utilization: float = 0.85
    health_timeout_s: int = 900
    structured_outputs: dict[str, Any] | None = field(
        default_factory=lambda: {"backend": "xgrammar", "disable_any_whitespace": True}
    )
    extra_vllm_args: tuple[str, ...] = ()


def parse_extraction_config(section: Mapping[str, Any], where: str) -> ExtractionConfig:
    reject_unknown_keys(section, EXTRACTION_KEYS, where)
    defaults = ExtractionConfig()
    port = optional(section, "port", int, defaults.port, where)
    workers = optional(section, "workers", int, defaults.workers, where)
    gpu = optional(
        section, "gpu_memory_utilization", (int, float), defaults.gpu_memory_utilization, where
    )
    timeout = optional(section, "health_timeout_s", int, defaults.health_timeout_s, where)
    if port < 1 or workers < 1 or timeout < 1 or not 0 < gpu <= 1:
        raise ConfigError(
            f"{where}: port, workers, and health_timeout_s must be >= 1 "
            "and gpu_memory_utilization in (0, 1]."
        )
    structured = section.get("structured_outputs", defaults.structured_outputs)
    if structured is not None and not isinstance(structured, dict):
        raise ConfigError(f"{where}.structured_outputs must be an object or null.")
    extra = section.get("extra_vllm_args", [])
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ConfigError(f"{where}.extra_vllm_args must be a list of strings.")
    return ExtractionConfig(
        port=port,
        workers=workers,
        gpu_memory_utilization=float(gpu),
        health_timeout_s=timeout,
        structured_outputs=dict(structured) if structured is not None else None,
        extra_vllm_args=tuple(extra),
    )


def extraction_dict(extraction: ExtractionConfig) -> dict[str, Any]:
    return {
        "port": extraction.port,
        "workers": extraction.workers,
        "gpu_memory_utilization": extraction.gpu_memory_utilization,
        "health_timeout_s": extraction.health_timeout_s,
        "structured_outputs": extraction.structured_outputs,
        "extra_vllm_args": list(extraction.extra_vllm_args),
    }
