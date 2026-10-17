"""The benchmark families a launch plan can name and how each one expands into runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.launcher import memory_family, rag_family, throughput_family

ExpandRuns = Callable[[dict[str, Any], dict[str, Any], RuntimeConfig, str, str, str], list[dict]]
ValidateRun = Callable[[dict[str, Any], RuntimeConfig, str], None]
PinPaths = Callable[[dict[str, Any], RuntimeConfig], None]


@dataclass(slots=True, frozen=True)
class BenchmarkFamily:
    name: str
    benchmark: str
    stages: frozenset[str]
    expand: ExpandRuns
    validate_run: ValidateRun
    pin_resolved_paths: PinPaths


FAMILIES: dict[str, BenchmarkFamily] = {
    "memory": BenchmarkFamily(
        name="memory",
        benchmark="memory-accuracy-latency",
        stages=frozenset(
            {"run", "judge", "extract", "check-catalogs", "preembed", "precompute-kv"}
        ),
        expand=memory_family.expand,
        validate_run=memory_family.validate_run,
        pin_resolved_paths=memory_family.pin_resolved_paths,
    ),
    "throughput": BenchmarkFamily(
        name="throughput",
        benchmark="memory-throughput",
        stages=frozenset({"run"}),
        expand=throughput_family.expand,
        validate_run=throughput_family.validate_run,
        pin_resolved_paths=throughput_family.pin_resolved_paths,
    ),
    "rag": BenchmarkFamily(
        name="rag",
        benchmark="rag",
        stages=frozenset({"run", "judge", "prepare", "estimate", "preembed", "precompute-kv"}),
        expand=rag_family.expand,
        validate_run=rag_family.validate_run,
        pin_resolved_paths=rag_family.pin_resolved_paths,
    ),
}


def family_of_config(data: dict[str, Any]) -> str:
    """The family whose configs carry this ``benchmark`` value, or the empty string."""
    benchmark = data.get("benchmark")
    for family in FAMILIES.values():
        if family.benchmark == benchmark:
            return family.name
    return ""
