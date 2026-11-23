from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.config import RuntimeConfig, slug, stamp_now
from benchmarks.common.json_fields import ConfigError, expand_path, load_json_object
from benchmarks.common.launcher.families import FAMILIES, BenchmarkFamily
from benchmarks.common.launcher.plans import (
    expand_runs,
    is_saved_manifest,
    load_launch_plan,
    references_saved_manifest,
    validate_plan,
    validate_stamp,
)


def materialize_plan(path: Path, runtime: RuntimeConfig, stage: str) -> tuple[dict[str, Any], Path]:
    plan = load_launch_plan(path)
    validate_plan(plan)
    if references_saved_manifest(plan):
        return _referenced_manifest(plan, runtime)
    if is_saved_manifest(plan):
        return _saved_manifest(plan, path)
    return _new_launch_folder(plan, runtime, stage)


def _referenced_manifest(
    plan: dict[str, Any], runtime: RuntimeConfig
) -> tuple[dict[str, Any], Path]:
    manifest_path = expand_path(plan["manifest"], root=runtime.root)
    manifest = load_launch_plan(manifest_path)
    validate_plan(manifest)
    if not is_saved_manifest(manifest) or manifest["family"] != plan["family"]:
        raise ConfigError(
            "launch.manifest must reference a saved manifest for the same benchmark family."
        )
    return _saved_manifest(manifest, manifest_path)


def _saved_manifest(manifest: dict[str, Any], manifest_path: Path) -> tuple[dict[str, Any], Path]:
    for run_config_path in manifest["run_configs"]:
        if not Path(run_config_path).is_file():
            raise ConfigError(f"Saved run config not found: {run_config_path}")
    return manifest, manifest_path.resolve()


def _new_launch_folder(
    plan: dict[str, Any], runtime: RuntimeConfig, stage: str
) -> tuple[dict[str, Any], Path]:
    stamp = plan.get("stamp") or stamp_now()
    validate_stamp(stamp)
    runs = expand_runs(plan, runtime, stage, stamp)

    name = plan.get("name") or plan["family"]
    folder = runtime.layout.results_root / "launches" / f"{slug(name)}-{stage}-{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    runtime_snapshot_path = _save_runtime_snapshot(runtime, folder)
    run_config_paths = _save_runs(runs, FAMILIES[plan["family"]], runtime, folder)
    manifest = {
        "family": plan["family"],
        "name": name,
        "stage": stage,
        "runtime_config": str(runtime_snapshot_path),
        "run_configs": [str(run_config_path) for run_config_path in run_config_paths],
    }
    manifest_path = folder / "manifest.json"
    atomic_write_json(manifest_path, manifest, indent=2)
    return manifest, manifest_path


def _save_runtime_snapshot(runtime: RuntimeConfig, folder: Path) -> Path:
    snapshot_path = folder / "runtime.json"
    atomic_write_json(snapshot_path, load_json_object(runtime.path), indent=2)
    return snapshot_path


def _save_runs(
    runs: list[dict[str, Any]], family: BenchmarkFamily, runtime: RuntimeConfig, folder: Path
) -> list[Path]:
    run_config_paths = []
    for run in runs:
        family.pin_to_runtime(run, runtime)
        run_config_path = folder / "runs" / f"{run['run_id']}.json"
        atomic_write_json(run_config_path, run, indent=2)
        run_config_paths.append(run_config_path)
    return run_config_paths
