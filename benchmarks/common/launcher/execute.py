"""Run every job of a launch manifest in order, and the ``launch`` entry point."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.config import RuntimeConfig, load_runtime_config
from benchmarks.common.environment import runtime_path_for_config
from benchmarks.common.json_fields import load_json_object, optional
from benchmarks.common.launcher.extraction import extraction_command, run_extraction
from benchmarks.common.launcher.manifest import materialize_plan
from benchmarks.common.launcher.plans import (
    job_stage,
    load_launch_plan,
    validate_plan,
    validate_run,
    validate_stage,
)
from benchmarks.common.launcher.processes import run_job
from benchmarks.common.launcher.rag_prepare import prepare_rag
from benchmarks.rag.config import RagBenchConfig


def launch(config_path: Path) -> int:
    """Execute the stage, runtime, and dry-run policy authored in a JSON plan."""
    config_path = Path(config_path)
    source = load_launch_plan(config_path)
    validate_plan(source)
    stage = source.get("stage") or "run"
    validate_stage(source["family"], stage)
    dry_run = optional(source, "dry_run", bool, False, "launch")
    runtime = load_runtime_config(runtime_path_for_config(config_path))
    runtime.apply_environment()
    if not dry_run:
        runtime.layout.create_directories()
    manifest, path = materialize_plan(config_path, runtime, stage)
    if "run_configs" not in source and "manifest" not in source:
        runtime = load_runtime_config(manifest["runtime_config"])
    return execute_manifest(manifest, path, runtime, stage, dry_run=dry_run)


def execute_manifest(
    manifest: dict[str, Any], path: Path, runtime: RuntimeConfig, stage: str, *, dry_run: bool
) -> int:
    """Validate every saved run, then run them one by one; returns 1 when any run failed."""
    family = manifest["family"]
    validate_stage(family, stage)
    for config_path in manifest["run_configs"]:
        data = load_json_object(config_path)
        validate_run(data, runtime, family, stage)
        if stage == "extract":
            extraction_command(data, runtime)
    failures: list[str] = []
    for index, config_path in enumerate(manifest["run_configs"], 1):
        data = load_json_object(config_path)
        if stage == "judge":
            config_path = _write_judge_input(path, config_path, data)
        job = {
            "family": family,
            "config": config_path,
            "runtime_config": str(runtime.path),
            "stage": job_stage(stage),
        }
        print(f"[{index}/{len(manifest['run_configs'])}] {data['run_id']}", flush=True)
        _announce(stage, data, runtime, job)
        if dry_run:
            continue
        log_file = path.parent / "logs" / f"{data['run_id']}.{stage}.log"
        if _run_stage(stage, family, data, runtime, job, log_file):
            failures.append(data["run_id"])
    if failures:
        print(f"Failed runs: {', '.join(failures)}", file=sys.stderr)
    print(f"Launch manifest: {path}")
    return int(bool(failures))


def _write_judge_input(manifest_path: Path, config_path: str, data: dict[str, Any]) -> str:
    """Save a judge-enabled copy of the run config beside the manifest."""
    data["skip_judge"] = False
    judge_path = manifest_path.parent / "judge-inputs" / Path(config_path).name
    atomic_write_json(judge_path, data, indent=2)
    return str(judge_path)


def _announce(
    stage: str, data: dict[str, Any], runtime: RuntimeConfig, job: dict[str, Any]
) -> None:
    if stage == "extract":
        server_command, _ = extraction_command(data, runtime)
        print(shlex.join(server_command), flush=True)
    if stage == "prepare":
        config = RagBenchConfig.from_dict(data, runtime=runtime)
        print(f"Prepare dataset {config.dataset_name} in {config.data_dir}")
    else:
        print(json.dumps(job), flush=True)


def _run_stage(
    stage: str,
    family: str,
    data: dict[str, Any],
    runtime: RuntimeConfig,
    job: dict[str, Any],
    log_file: Path,
) -> int:
    if stage == "prepare":
        prepare_rag(data, runtime)
        return 0
    if stage == "run" and family == "memory":
        guard = {**job, "stage": "check-catalogs"}
        if run_job(guard, log_file.with_suffix(".catalogs.log")):
            return 1
    if stage == "extract":
        return run_extraction(data, runtime, job, log_file)
    return run_job(job, log_file)
