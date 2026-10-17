"""Throughput benchmark runs: one run per config, with the sweep's KV store settings applied."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError, expand_path
from benchmarks.memory.throughput.config import ThroughputConfig

SWEEP_KV_SETTINGS = (("kv_store_backend", "store_backend"), ("kv_staging_slots", "staging_slots"))


def expand(
    data: dict[str, Any],
    plan: dict[str, Any],
    runtime: RuntimeConfig,
    stage: str,
    stamp: str,
    where: str,
) -> list[dict[str, Any]]:
    del stage
    ThroughputConfig.from_dict(data, runtime=runtime)
    sweep = _sweep(data, plan.get("sweep"), where)
    run = deepcopy(data)
    if "conditions" in sweep:
        run["conditions"] = sweep["conditions"]
    for source, target in SWEEP_KV_SETTINGS:
        if source in sweep:
            run["inferscale"].setdefault("kv", {})[target] = sweep[source]
    if not run.get("run_id"):
        prefix = sweep.get("run_prefix", "throughput")
        run["run_id"] = f"{prefix}-{stamp}-{runtime.model_label(data['model'])}"
    ThroughputConfig.from_dict(run, runtime=runtime)
    return [run]


def _sweep(data: dict[str, Any], sweep_name: str | None, where: str) -> dict[str, Any]:
    if not sweep_name:
        return {}
    sweep = data.get("sweeps", {}).get(sweep_name, {})
    if not sweep:
        raise ConfigError(f"No throughput sweep {sweep_name!r} in {where}.")
    return sweep


def validate_run(data: dict[str, Any], runtime: RuntimeConfig, stage: str) -> None:
    del stage
    ThroughputConfig.from_dict(data, runtime=runtime)


def pin_resolved_paths(run: dict[str, Any], runtime: RuntimeConfig) -> None:
    run["model"] = runtime.resolve_model(run["model"])
    run["dataset_path"] = str(
        expand_path(run.get("dataset_path", "data/locomo10.json"), root=runtime.root)
    )
    if run.get("results_dir"):
        run["results_dir"] = str(expand_path(run["results_dir"], root=runtime.root))
    else:
        run["results_dir"] = str(runtime.layout.results_root)
