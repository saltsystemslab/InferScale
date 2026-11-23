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
    gpu_memory_utilization = optional(
        section, "gpu_memory_utilization", (int, float), defaults.gpu_memory_utilization, where
    )
    health_timeout_s = optional(section, "health_timeout_s", int, defaults.health_timeout_s, where)
    if port < 1 or workers < 1 or health_timeout_s < 1 or not 0 < gpu_memory_utilization <= 1:
        raise ConfigError(
            f"{where}: port, workers, and health_timeout_s must be >= 1 "
            "and gpu_memory_utilization in (0, 1]."
        )
    structured_outputs = section.get("structured_outputs", defaults.structured_outputs)
    if structured_outputs is not None and not isinstance(structured_outputs, dict):
        raise ConfigError(f"{where}.structured_outputs must be an object or null.")
    extra_vllm_args = section.get("extra_vllm_args", [])
    if not _is_list_of_strings(extra_vllm_args):
        raise ConfigError(f"{where}.extra_vllm_args must be a list of strings.")
    return ExtractionConfig(
        port=port,
        workers=workers,
        gpu_memory_utilization=float(gpu_memory_utilization),
        health_timeout_s=health_timeout_s,
        structured_outputs=dict(structured_outputs) if structured_outputs is not None else None,
        extra_vllm_args=tuple(extra_vllm_args),
    )


def _is_list_of_strings(candidate: Any) -> bool:
    return isinstance(candidate, list) and all(isinstance(entry, str) for entry in candidate)


def extraction_dict(extraction: ExtractionConfig) -> dict[str, Any]:
    return {
        "port": extraction.port,
        "workers": extraction.workers,
        "gpu_memory_utilization": extraction.gpu_memory_utilization,
        "health_timeout_s": extraction.health_timeout_s,
        "structured_outputs": extraction.structured_outputs,
        "extra_vllm_args": list(extraction.extra_vllm_args),
    }
