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
from benchmarks.common.launcher.extraction import validate_extraction_server
from benchmarks.common.launcher.families import FAMILIES, BenchmarkFamily, family_of_config

WHERE = "launch"
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
DEFAULT_STAGE = "run"
PREEMBED_SETUP_STAGES = frozenset({"extract", "prepare"})
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
OPTIONAL_TEXT_KEYS = ("name", "sweep", "stamp", "runtime_config", "stage", "manifest")


def benchmark_stage(stage: str) -> str:
    return "preembed" if stage in PREEMBED_SETUP_STAGES else stage


def validate_stage(family: str, stage: str) -> None:
    if family not in FAMILIES or stage not in FAMILIES[family].stages:
        raise ConfigError(f"Stage {stage!r} is not supported by {family} launch plans.")


def load_launch_plan(path: Path) -> dict[str, Any]:
    plan_or_benchmark_config = load_json_object(path)
    family = family_of_config(plan_or_benchmark_config)
    if family is None:
        return plan_or_benchmark_config
    return _plan_launching_one_config(family, path)


def _plan_launching_one_config(family: str, config_path: Path) -> dict[str, Any]:
    return {
        "family": family,
        "name": f"{family}-{slug(config_path.stem)}",
        "configs": [str(config_path.resolve())],
    }


def references_saved_manifest(plan: dict[str, Any]) -> bool:
    return "manifest" in plan


def is_saved_manifest(plan: dict[str, Any]) -> bool:
    return "run_configs" in plan


def starts_new_launch(plan: dict[str, Any]) -> bool:
    return not references_saved_manifest(plan) and not is_saved_manifest(plan)


def validate_plan(plan: dict[str, Any]) -> None:
    reject_unknown_keys(plan, PLAN_KEYS, WHERE)
    _validate_family(plan)
    _validate_optional_text(plan)
    _validate_stage_name(plan)
    _validate_dry_run_flag(plan)
    if references_saved_manifest(plan):
        _validate_manifest_reference(plan)
    elif is_saved_manifest(plan):
        _validate_saved_manifest(plan)
    else:
        _validate_config_paths(plan, "configs")


def _validate_family(plan: dict[str, Any]) -> None:
    family = require(plan, "family", str, WHERE)
    if family not in FAMILIES:
        raise ConfigError(f"Unknown launch family {family!r}.")


def _validate_optional_text(plan: dict[str, Any]) -> None:
    for key in OPTIONAL_TEXT_KEYS:
        text = optional(plan, key, str, None, WHERE)
        if text is not None and not text.strip():
            raise ConfigError(f"launch.{key} must not be empty.")


def _validate_stage_name(plan: dict[str, Any]) -> None:
    stage = plan.get("stage")
    if stage is not None and stage not in STAGES:
        raise ConfigError(f"Unknown launch stage {stage!r}.")


def _validate_dry_run_flag(plan: dict[str, Any]) -> None:
    optional(plan, "dry_run", bool, False, WHERE)


def _validate_manifest_reference(plan: dict[str, Any]) -> None:
    require(plan, "manifest", str, WHERE)
    if any(key in plan for key in ("configs", "sweep", "run_configs")):
        raise ConfigError(
            "A launch manifest reference cannot also specify configs, run_configs or a sweep."
        )


def _validate_saved_manifest(plan: dict[str, Any]) -> None:
    if "configs" in plan or "sweep" in plan:
        raise ConfigError("Saved launch manifests cannot also specify configs or a sweep.")
    require(plan, "runtime_config", str, WHERE)
    _validate_config_paths(plan, "run_configs")


def _validate_config_paths(plan: dict[str, Any], key: str) -> None:
    paths = require(plan, key, list, WHERE)
    if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ConfigError(f"launch.{key} must be a non-empty list of JSON config paths.")


def validate_stamp(stamp: Any) -> None:
    if not _is_safe_file_name(stamp):
        raise ConfigError(
            "launch.stamp must contain only letters, numbers, dots, dashes and underscores."
        )


def _is_safe_file_name(value: Any) -> bool:
    return isinstance(value, str) and slug(value) == value.lower()


def expand_runs(
    plan: dict[str, Any], runtime: RuntimeConfig, stage: str, stamp: str
) -> list[dict[str, Any]]:
    validate_plan(plan)
    family = FAMILIES[plan["family"]]
    validate_stage(family.name, stage)
    if "configs" not in plan:
        raise ConfigError("launch.configs must be a non-empty list of JSON config paths.")

    sweep_name = plan.get("sweep") if stage == "run" else None
    runs: list[dict[str, Any]] = []
    for config_path in plan["configs"]:
        authored = _load_authored_config(config_path, runtime, family)
        if stage == "judge" and not authored.get("run_id"):
            raise ConfigError(
                "Judge plans require an existing run_id; select the saved launch manifest."
            )
        runs += family.expand(
            authored,
            config_path=config_path,
            runtime=runtime,
            stage=stage,
            stamp=stamp,
            sweep_name=sweep_name,
        )

    _validate_run_ids(runs)
    for run in runs:
        validate_run(run, runtime, family.name, stage)
    return runs


def _load_authored_config(
    config_path: str, runtime: RuntimeConfig, family: BenchmarkFamily
) -> dict[str, Any]:
    authored = load_json_object(expand_path(config_path, root=runtime.root))
    if family_of_config(authored) != family.name:
        raise ConfigError(f"{config_path} does not contain a {family.name} benchmark config.")
    return authored


def _validate_run_ids(runs: list[dict[str, Any]]) -> None:
    run_ids = [run["run_id"] for run in runs]
    if not all(_is_safe_file_name(run_id) for run_id in run_ids):
        raise ConfigError(
            "Run IDs must contain only letters, numbers, dots, dashes and underscores."
        )
    if len(run_ids) != len(set(run_ids)):
        raise ConfigError(
            "Launch plan produces duplicate run IDs; select unique configs or remove fixed "
            "run_id values."
        )


def validate_run(run: dict[str, Any], runtime: RuntimeConfig, family: str, stage: str) -> None:
    if family_of_config(run) != family:
        raise ConfigError(f"Saved input does not match launch family {family}.")
    if not run.get("run_id"):
        raise ConfigError("Materialized benchmark inputs require an explicit run_id.")
    if stage == "judge":
        run = with_judging_enabled(run)
    FAMILIES[family].validate_run(run, runtime, benchmark_stage(stage))
    if stage == "extract":
        validate_extraction_server(run, runtime)


def with_judging_enabled(run: dict[str, Any]) -> dict[str, Any]:
    return {**run, "skip_judge": False}
