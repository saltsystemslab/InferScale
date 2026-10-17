"""Launch plans: the JSON files under configs/launch, their stages, and their expansion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.common.config import RuntimeConfig, slug
from benchmarks.common.json_fields import (
    ConfigError,
    expand_path,
    load_json_object,
    optional,
    reject_unknown_keys,
    require,
)
from benchmarks.common.launcher.families import FAMILIES, family_of_config

STAGES = (
    "run",
    "extract",
    "check-catalogs",
    "preembed",
    "precompute-kv",
    "prepare",
    "estimate",
    "judge",
)
PLAN_KEYS = (
    "family",
    "name",
    "configs",
    "sweep",
    "stamp",
    "run_configs",
    "runtime_config",
    "stage",
    "dry_run",
    "manifest",
)
# The extract and prepare stages run the preembed job once their own setup (a
# local extraction server, a dataset download) is done, so the benchmark
# process is sent the preembed stage.
JOB_STAGE_FOR = {"extract": "preembed", "prepare": "preembed"}


def job_stage(stage: str) -> str:
    return JOB_STAGE_FOR.get(stage, stage)


def validate_stage(family: str, stage: str) -> None:
    allowed = FAMILIES[family].stages if family in FAMILIES else frozenset()
    if stage not in allowed:
        raise ConfigError(f"Stage {stage!r} is not supported by {family} launch plans.")


def validate_plan(plan: dict[str, Any]) -> None:
    reject_unknown_keys(plan, PLAN_KEYS, "launch")
    family = require(plan, "family", str, "launch")
    if family not in FAMILIES:
        raise ConfigError(f"Unknown launch family {family!r}.")
    for name in ("name", "sweep", "stamp", "runtime_config", "stage", "manifest"):
        value = optional(plan, name, str, None, "launch")
        if value is not None and not value.strip():
            raise ConfigError(f"launch.{name} must not be empty.")
    if plan.get("stage") is not None and plan["stage"] not in STAGES:
        raise ConfigError(f"Unknown launch stage {plan['stage']!r}.")
    optional(plan, "dry_run", bool, False, "launch")
    if "manifest" in plan:
        require(plan, "manifest", str, "launch")
        if any(key in plan for key in ("configs", "sweep", "run_configs")):
            raise ConfigError(
                "A launch manifest reference cannot also specify configs, run_configs or a sweep."
            )
        return
    if "run_configs" in plan:
        if "configs" in plan or "sweep" in plan:
            raise ConfigError("Saved launch manifests cannot also specify configs or a sweep.")
        require(plan, "runtime_config", str, "launch")
        key = "run_configs"
    else:
        key = "configs"
    paths = require(plan, key, list, "launch")
    if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ConfigError(f"launch.{key} must be a non-empty list of JSON config paths.")


def load_launch_plan(path: Path) -> dict[str, Any]:
    """A launch plan, or a single benchmark config wrapped as a one-run plan."""
    data = load_json_object(path)
    family = family_of_config(data)
    if family:
        return {
            "family": family,
            "name": f"{family}-{slug(path.stem)}",
            "configs": [str(path.resolve())],
        }
    return data


def validate_stamp(stamp: Any) -> None:
    if not isinstance(stamp, str) or slug(stamp) != stamp.lower():
        raise ConfigError(
            "launch.stamp must contain only letters, numbers, dots, dashes and underscores."
        )


def validate_run(data: dict[str, Any], runtime: RuntimeConfig, family: str, stage: str) -> None:
    """Check one materialized run before any benchmark process is started."""
    if family_of_config(data) != family:
        raise ConfigError(f"Saved input does not match launch family {family}.")
    if not data.get("run_id"):
        raise ConfigError("Materialized benchmark inputs require an explicit run_id.")
    if stage == "judge":
        data = {**data, "skip_judge": False}
    FAMILIES[family].validate_run(data, runtime, job_stage(stage))


def expand_runs(
    plan: dict[str, Any], runtime: RuntimeConfig, stage: str, stamp: str
) -> list[dict[str, Any]]:
    """Materialize the runs a plan describes, without changing the authored configs."""
    validate_plan(plan)
    family = FAMILIES[plan["family"]]
    validate_stage(family.name, stage)
    paths = plan.get("configs")
    if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
        raise ConfigError("launch.configs must be a non-empty list of JSON config paths.")
    runs: list[dict[str, Any]] = []
    for path in paths:
        data = load_json_object(expand_path(path, root=runtime.root))
        if family_of_config(data) != family.name:
            raise ConfigError(f"{path} does not contain a {family.name} benchmark config.")
        runs.extend(family.expand(data, plan, runtime, stage, stamp, str(path)))
    _validate_run_ids(runs)
    for run in runs:
        validate_run(run, runtime, family.name, stage)
    return runs


def _validate_run_ids(runs: list[dict[str, Any]]) -> None:
    ids = [run["run_id"] for run in runs]
    if any(not isinstance(item, str) or not item or slug(item) != item.lower() for item in ids):
        raise ConfigError(
            "Run IDs must contain only letters, numbers, dots, dashes and underscores."
        )
    if len(ids) != len(set(ids)):
        raise ConfigError(
            "Launch plan produces duplicate run IDs; select unique configs or remove fixed "
            "run_id values."
        )
