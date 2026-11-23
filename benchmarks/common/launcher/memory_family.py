from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.memory.config import MemoryRunConfig, validate_memory_stage

EXTRACTION_VECTOR_BACKEND = "qdrant"


def expand(
    authored: dict[str, Any],
    *,
    config_path: str,
    runtime: RuntimeConfig,
    stage: str,
    stamp: str,
    sweep_name: str | None,
) -> list[dict[str, Any]]:
    config = MemoryRunConfig.from_dict(authored, runtime=runtime)
    if sweep_name:
        return _run_per_sweep_cell(config, authored, sweep_name, stamp)
    if stage == "precompute-kv":
        return _run_per_context_window(config, authored, stamp)
    return [_single_run(config, authored, stage, stamp)]


def _run_per_sweep_cell(
    config: MemoryRunConfig, authored: dict[str, Any], sweep_name: str, stamp: str
) -> list[dict[str, Any]]:
    runs = []
    for cell in config.sweep_cells(sweep_name):
        run = cell.materialize(authored)
        run["run_id"] = cell.run_id(config.label, config.max_samples, stamp)
        run["skip_judge"] = True
        runs.append(run)
    return runs


def _run_per_context_window(
    config: MemoryRunConfig, authored: dict[str, Any], stamp: str
) -> list[dict[str, Any]]:
    runs = []
    for context_window in config.context_windows():
        run = deepcopy(authored)
        run["context_window"] = context_window
        run["run_id"] = f"precompute-{config.label}-s{context_window}-{stamp}"
        runs.append(run)
    return runs


def _single_run(
    config: MemoryRunConfig, authored: dict[str, Any], stage: str, stamp: str
) -> dict[str, Any]:
    run = deepcopy(authored)
    if stage == "extract":
        run["preembed_workers"] = config.extraction.workers
        run["vector_backend"] = EXTRACTION_VECTOR_BACKEND
    if not run.get("run_id"):
        run["run_id"] = f"{stage}-{config.label}-{stamp}"
    return run


def validate_run(run: dict[str, Any], runtime: RuntimeConfig, stage: str) -> None:
    config = MemoryRunConfig.from_dict(run, runtime=runtime)
    validate_memory_stage(config, stage, has_run_id=True)


def pin_to_runtime(run: dict[str, Any], runtime: RuntimeConfig) -> None:
    config = MemoryRunConfig.from_dict(run, runtime=runtime)
    run["model"] = config.model
    run["dataset_path"] = str(config.dataset_path)
    run["results_dir"] = str(config.results_dir)
