from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.launcher import memory_family, rag_family, throughput_family


class ExpandRuns(Protocol):
    def __call__(
        self,
        authored: dict[str, Any],
        *,
        config_path: str,
        runtime: RuntimeConfig,
        stage: str,
        stamp: str,
        sweep_name: str | None,
    ) -> list[dict[str, Any]]: ...


ValidateRun = Callable[[dict[str, Any], RuntimeConfig, str], None]
PinToRuntime = Callable[[dict[str, Any], RuntimeConfig], None]


@dataclass(slots=True, frozen=True)
class BenchmarkFamily:
    name: str
    benchmark: str
    stages: frozenset[str]
    expand: ExpandRuns
    validate_run: ValidateRun
    pin_to_runtime: PinToRuntime


FAMILIES: dict[str, BenchmarkFamily] = {
    "memory": BenchmarkFamily(
        name="memory",
        benchmark="memory-accuracy-latency",
        stages=frozenset(
            {"run", "judge", "extract", "check-catalogs", "preembed", "precompute-kv"}
        ),
        expand=memory_family.expand,
        validate_run=memory_family.validate_run,
        pin_to_runtime=memory_family.pin_to_runtime,
    ),
    "throughput": BenchmarkFamily(
        name="throughput",
        benchmark="memory-throughput",
        stages=frozenset({"run"}),
        expand=throughput_family.expand,
        validate_run=throughput_family.validate_run,
        pin_to_runtime=throughput_family.pin_to_runtime,
    ),
    "rag": BenchmarkFamily(
        name="rag",
        benchmark="rag",
        stages=frozenset({"run", "judge", "prepare", "estimate", "preembed", "precompute-kv"}),
        expand=rag_family.expand,
        validate_run=rag_family.validate_run,
        pin_to_runtime=rag_family.pin_to_runtime,
    ),
}


def family_of_config(config: dict[str, Any]) -> str | None:
    benchmark = config.get("benchmark")
    for family in FAMILIES.values():
        if family.benchmark == benchmark:
            return family.name
    return None
