"""Launch folders: the expanded run configs and the manifest a later judge pass replays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.config import RuntimeConfig, slug, stamp_now
from benchmarks.common.json_fields import ConfigError, expand_path, load_json_object
from benchmarks.common.launcher.extraction import extraction_command
from benchmarks.common.launcher.families import FAMILIES
from benchmarks.common.launcher.plans import (
    expand_runs,
    load_launch_plan,
    validate_plan,
    validate_stamp,
)


def materialize_plan(path: Path, runtime: RuntimeConfig, stage: str) -> tuple[dict[str, Any], Path]:
    """Expand a plan into a launch folder, or resolve a saved manifest it references."""
    plan = load_launch_plan(path)
    validate_plan(plan)
    if "manifest" in plan:
        return materialize_plan(_saved_manifest_path(plan, runtime), runtime, stage)
    if "run_configs" in plan:
        _check_saved_run_configs(plan["run_configs"])
        return plan, path.resolve()
    stamp = plan.get("stamp") or stamp_now()
    validate_stamp(stamp)
    runs = expand_runs(plan, runtime, stage, stamp)
    if stage == "extract":
        for run in runs:
            extraction_command(run, runtime)
    return _write_launch_folder(plan, runs, runtime, stage, stamp)


def _saved_manifest_path(plan: dict[str, Any], runtime: RuntimeConfig) -> Path:
    saved_path = expand_path(plan["manifest"], root=runtime.root)
    saved = load_launch_plan(saved_path)
    validate_plan(saved)
    if "run_configs" not in saved or saved["family"] != plan["family"]:
        raise ConfigError(
            "launch.manifest must reference a saved manifest for the same benchmark family."
        )
    return saved_path


def _check_saved_run_configs(configs: Any) -> None:
    if not isinstance(configs, list) or not configs or any(not isinstance(c, str) for c in configs):
        raise ConfigError("Saved launch manifest run_configs must be a non-empty list of paths.")
    for item in configs:
        if not Path(item).is_file():
            raise ConfigError(f"Saved run config not found: {item}")


def _write_launch_folder(
    plan: dict[str, Any],
    runs: list[dict[str, Any]],
    runtime: RuntimeConfig,
    stage: str,
    stamp: str,
) -> tuple[dict[str, Any], Path]:
    family = FAMILIES[plan["family"]]
    name = plan.get("name") or plan["family"]
    folder = runtime.layout.results_root / "launches" / f"{slug(name)}-{stage}-{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    runtime_path = folder / "runtime.json"
    atomic_write_json(runtime_path, load_json_object(runtime.path), indent=2)
    manifest: dict[str, Any] = {
        "family": family.name,
        "name": name,
        "stage": stage,
        "runtime_config": str(runtime_path),
        "run_configs": [],
    }
    for run in runs:
        # Pin the answer model and the data and results locations so a later
        # judge pass reads the same files even when it selects another runtime.
        family.pin_resolved_paths(run, runtime)
        target = folder / "runs" / f"{run['run_id']}.json"
        atomic_write_json(target, run, indent=2)
        manifest["run_configs"].append(str(target))
    manifest_path = folder / "manifest.json"
    atomic_write_json(manifest_path, manifest, indent=2)
    return manifest, manifest_path
