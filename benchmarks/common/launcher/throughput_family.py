from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError, expand_path
from benchmarks.memory.throughput.config import DEFAULT_DATASET_PATH, ThroughputConfig

DEFAULT_RUN_PREFIX = "throughput"
KV_SETTING_OF_SWEEP_KEY = {
    "kv_store_backend": "store_backend",
    "kv_staging_slots": "staging_slots",
}


def expand(
    authored: dict[str, Any],
    *,
    config_path: str,
    runtime: RuntimeConfig,
    stage: str,
    stamp: str,
    sweep_name: str | None,
) -> list[dict[str, Any]]:
    _require_valid_config(authored, runtime)
    sweep = _sweep(authored, sweep_name, config_path)
    run = deepcopy(authored)
    if "conditions" in sweep:
        run["conditions"] = sweep["conditions"]
    for sweep_key, kv_setting in KV_SETTING_OF_SWEEP_KEY.items():
        if sweep_key in sweep:
            run["inferscale"].setdefault("kv", {})[kv_setting] = sweep[sweep_key]
    if not run.get("run_id"):
        run_prefix = sweep.get("run_prefix", DEFAULT_RUN_PREFIX)
        run["run_id"] = f"{run_prefix}-{stamp}-{runtime.model_label(authored['model'])}"
    return [run]


def _require_valid_config(config: dict[str, Any], runtime: RuntimeConfig) -> None:
    ThroughputConfig.from_dict(config, runtime=runtime)


def _sweep(authored: dict[str, Any], sweep_name: str | None, config_path: str) -> dict[str, Any]:
    if not sweep_name:
        return {}
    sweep = authored.get("sweeps", {}).get(sweep_name, {})
    if not sweep:
        raise ConfigError(f"No throughput sweep {sweep_name!r} in {config_path}.")
    return sweep


def validate_run(run: dict[str, Any], runtime: RuntimeConfig, stage: str) -> None:
    _require_valid_config(run, runtime)


def pin_to_runtime(run: dict[str, Any], runtime: RuntimeConfig) -> None:
    run["model"] = runtime.resolve_model(run["model"])
    dataset_path = run.get("dataset_path", DEFAULT_DATASET_PATH)
    run["dataset_path"] = str(expand_path(dataset_path, root=runtime.root))
    if run.get("results_dir"):
        run["results_dir"] = str(expand_path(run["results_dir"], root=runtime.root))
    else:
        run["results_dir"] = str(runtime.layout.results_root)
