from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.config import RuntimeConfig, load_runtime_config
from benchmarks.common.environment import runtime_path_for_config
from benchmarks.common.json_fields import load_json_object, optional
from benchmarks.common.launcher.extraction import extraction_command, run_extraction
from benchmarks.common.launcher.manifest import materialize_plan
from benchmarks.common.launcher.plans import (
    DEFAULT_STAGE,
    benchmark_stage,
    load_launch_plan,
    starts_new_launch,
    validate_plan,
    validate_run,
    validate_stage,
    with_judging_enabled,
)
from benchmarks.common.launcher.processes import run_job
from benchmarks.common.launcher.rag_prepare import prepare_rag
from benchmarks.common.logging import configure_logging
from benchmarks.rag.config import RagBenchConfig


def launch(plan_path: str | Path) -> int:
    configure_logging()
    plan_path = Path(plan_path)
    plan = load_launch_plan(plan_path)
    validate_plan(plan)
    stage = plan.get("stage") or DEFAULT_STAGE
    validate_stage(plan["family"], stage)
    dry_run = optional(plan, "dry_run", bool, False, "launch")

    runtime = load_runtime_config(runtime_path_for_config(plan_path))
    runtime.apply_environment()
    if not dry_run:
        runtime.layout.create_directories()

    manifest, manifest_path = materialize_plan(plan_path, runtime, stage)
    if starts_new_launch(plan):
        runtime_snapshot_path = manifest["runtime_config"]
        runtime = load_runtime_config(runtime_snapshot_path)
    return execute_manifest(manifest, manifest_path, runtime, stage, dry_run=dry_run)


@dataclass(frozen=True, slots=True)
class _StageExecution:
    family: str
    stage: str
    runtime: RuntimeConfig
    launch_folder: Path


def execute_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    runtime: RuntimeConfig,
    stage: str,
    *,
    dry_run: bool,
) -> int:
    family = manifest["family"]
    validate_stage(family, stage)
    execution = _StageExecution(
        family=family, stage=stage, runtime=runtime, launch_folder=manifest_path.parent
    )
    run_config_paths = manifest["run_configs"]
    _validate_every_run(execution, run_config_paths)

    failed_run_ids: list[str] = []
    for run_number, run_config_path in enumerate(run_config_paths, start=1):
        run_config = load_json_object(run_config_path)
        run_id = run_config["run_id"]
        logger.info("[{}/{}] {}", run_number, len(run_config_paths), run_id)

        job = _build_job(execution, run_config, run_config_path)
        _announce_run(execution, run_config, job)
        if dry_run:
            continue
        exit_code = _execute_run(execution, run_config, job)
        if exit_code != 0:
            failed_run_ids.append(run_id)

    if failed_run_ids:
        logger.error("Failed runs: {}", ", ".join(failed_run_ids))
    logger.info("Launch manifest: {}", manifest_path)
    return 1 if failed_run_ids else 0


def _validate_every_run(execution: _StageExecution, run_config_paths: list[str]) -> None:
    for run_config_path in run_config_paths:
        run_config = load_json_object(run_config_path)
        validate_run(run_config, execution.runtime, execution.family, execution.stage)


def _build_job(
    execution: _StageExecution, run_config: dict[str, Any], run_config_path: str
) -> dict[str, Any]:
    job_config_path = run_config_path
    if execution.stage == "judge":
        job_config_path = _write_judge_enabled_copy(
            run_config, run_config_path, execution.launch_folder
        )
    return {
        "family": execution.family,
        "config": job_config_path,
        "runtime_config": str(execution.runtime.path),
        "stage": benchmark_stage(execution.stage),
    }


def _write_judge_enabled_copy(
    run_config: dict[str, Any], run_config_path: str, launch_folder: Path
) -> str:
    judge_config_path = launch_folder / "judge-inputs" / Path(run_config_path).name
    atomic_write_json(judge_config_path, with_judging_enabled(run_config), indent=2)
    return str(judge_config_path)


def _announce_run(
    execution: _StageExecution, run_config: dict[str, Any], job: dict[str, Any]
) -> None:
    if execution.stage == "prepare":
        rag_config = RagBenchConfig.from_dict(run_config, runtime=execution.runtime)
        logger.info("Prepare dataset {} in {}", rag_config.dataset_name, rag_config.data_dir)
        return
    if execution.stage == "extract":
        server_command, _ = extraction_command(run_config, execution.runtime)
        logger.info(shlex.join(server_command))
    logger.info(json.dumps(job))


def _execute_run(
    execution: _StageExecution, run_config: dict[str, Any], job: dict[str, Any]
) -> int:
    if execution.stage == "prepare":
        prepare_rag(run_config, execution.runtime)
        return 0

    run_id = run_config["run_id"]
    log_file = execution.launch_folder / "logs" / f"{run_id}.{execution.stage}.log"
    if execution.stage == "extract":
        return run_extraction(run_config, execution.runtime, job, log_file)
    if False:
        catalog_check = {**job, "stage": "check-catalogs"}
        catalog_check_exit_code = run_job(catalog_check, log_file.with_suffix(".catalogs.log"))
        if catalog_check_exit_code != 0:
            return catalog_check_exit_code
    return run_job(job, log_file)
