"""Memory benchmark runs: sweep cells, one precompute run per window, or one stage run."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError
from benchmarks.memory.config import MemoryRunConfig, validate_memory_stage


def expand(
    data: dict[str, Any],
    plan: dict[str, Any],
    runtime: RuntimeConfig,
    stage: str,
    stamp: str,
    where: str,
) -> list[dict[str, Any]]:
    del where
    config = MemoryRunConfig.from_dict(data, runtime=runtime)
    if stage == "run" and plan.get("sweep"):
        runs = _sweep_runs(config, data, plan["sweep"], stamp)
    elif stage == "precompute-kv":
        runs = _precompute_runs(config, data, stamp)
    else:
        runs = [_stage_run(config, data, stage, stamp)]
    for run in runs:
        MemoryRunConfig.from_dict(run, runtime=runtime)
    return runs


def _sweep_runs(
    config: MemoryRunConfig, data: dict[str, Any], sweep_name: str, stamp: str
) -> list[dict[str, Any]]:
    runs = []
    for cell in config.sweep_cells(sweep_name):
        run = cell.materialize(data)
        run["run_id"] = cell.run_id(config.label, config.max_samples, stamp)
        run["skip_judge"] = True
        runs.append(run)
    return runs


def _precompute_runs(
    config: MemoryRunConfig, data: dict[str, Any], stamp: str
) -> list[dict[str, Any]]:
    runs = []
    for window in config.context_windows():
        run = deepcopy(data)
        run["context_window"] = window
        run["run_id"] = f"precompute-{config.label}-s{window}-{stamp}"
        runs.append(run)
    return runs


def _stage_run(
    config: MemoryRunConfig, data: dict[str, Any], stage: str, stamp: str
) -> dict[str, Any]:
    run = deepcopy(data)
    if stage == "extract":
        run["preembed_workers"] = config.extraction.workers
        run["vector_backend"] = "qdrant"
    if stage == "judge" and not run.get("run_id"):
        raise ConfigError(
            "Judge plans require an existing run_id; select the saved launch manifest."
        )
    if not run.get("run_id"):
        run["run_id"] = f"{stage}-{config.label}-{stamp}"
    return run


def validate_run(data: dict[str, Any], runtime: RuntimeConfig, stage: str) -> None:
    config = MemoryRunConfig.from_dict(data, runtime=runtime)
    validate_memory_stage(config, stage, has_run_id=True)


def pin_resolved_paths(run: dict[str, Any], runtime: RuntimeConfig) -> None:
    resolved = MemoryRunConfig.from_dict(run, runtime=runtime)
    run["model"] = resolved.model
    run["dataset_path"] = str(resolved.dataset_path)
    run["results_dir"] = str(resolved.results_dir)
