from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.files import write_json
from benchmarks.common.json_fields import ConfigError
from benchmarks.common.paths import RUN_CONFIG_FILE, project_root
from benchmarks.common.system import collect_system_metadata
from benchmarks.memory.throughput.config import (
    LMCACHE_CONDITION,
    WORKER_API_KEY_VARIABLE,
    ThroughputConfig,
    lmcache_config,
    user_counts_text,
)
from benchmarks.memory.throughput.reporting import (
    merge_result_rows,
    read_existing_results,
    validate_result_row,
    write_reports,
)

VOLATILE_CONFIG_KEYS = frozenset(
    {"conditions", "embedding_api_key", "local_store_dir", "log_level"}
)
REQUIRED_MODULES = {"vllm", "mem0"}
CONDITION_MODULES = {
    "kv_injection": ("torch", "transformers", "jasper"),
    "mem0_qdrant": ("qdrant_client",),
    "mem0_jasper": ("jasper",),
    LMCACHE_CONDITION: ("lmcache", "qdrant_client"),
}


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    condition: str
    user_counts: tuple[int, ...]
    output_path: Path


def worker_specs(config: ThroughputConfig) -> list[WorkerSpec]:
    specs: list[WorkerSpec] = []
    worker_dir = config.run_dir / "worker-results"
    for condition in config.conditions:
        if condition == "kv_injection":
            specs.extend(
                WorkerSpec(condition, (num_users,), worker_dir / f"{condition}-{num_users}u.json")
                for num_users in config.user_counts
            )
        else:
            specs.append(
                WorkerSpec(condition, config.user_counts, worker_dir / f"{condition}.json")
            )
    return specs


def build_worker_job(config: ThroughputConfig, spec: WorkerSpec) -> dict[str, Any]:
    return {
        "config_path": str(_config_snapshot_path(config).resolve()),
        "condition": spec.condition,
        "user_counts": list(spec.user_counts),
        "output_path": str(spec.output_path.resolve()),
    }


def run_throughput(config: ThroughputConfig, *, dry_run: bool = False) -> dict[str, Any] | None:
    specs = worker_specs(config)
    if dry_run:
        _announce_dry_run(config, specs)
        return None

    _validate_runtime_requirements(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_config(config)
    rows = read_existing_results(config.run_dir)
    write_json(_config_snapshot_path(config), config.to_jsonable())
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / "system.json", system_metadata)

    for worker_number, spec in enumerate(specs, start=1):
        logger.info(
            "[{}/{}] {} users={}",
            worker_number,
            len(specs),
            spec.condition,
            user_counts_text(spec.user_counts),
        )
        _run_worker(config, spec)
        rows = merge_result_rows(rows, _load_worker_results(spec))
        write_reports(config, rows, system_metadata=system_metadata)

    summary = write_reports(config, rows, system_metadata=system_metadata)
    logger.info("wrote {} throughput rows to {}", summary["row_count"], config.run_dir)
    return summary


def _config_snapshot_path(config: ThroughputConfig) -> Path:
    return config.run_dir / RUN_CONFIG_FILE


def _announce_dry_run(config: ThroughputConfig, specs: list[WorkerSpec]) -> None:
    logger.info("run directory: {}", config.run_dir)
    logger.info("model: {}", config.model)
    logger.info("user counts: {}", user_counts_text(config.user_counts))
    for spec in specs:
        logger.info(json.dumps(build_worker_job(config, spec)))


def _run_worker(config: ThroughputConfig, spec: WorkerSpec) -> None:
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "benchmarks.memory.throughput.worker"],
        input=json.dumps(build_worker_job(config, spec)),
        text=True,
        check=True,
        cwd=project_root(),
        env=_worker_environment(config),
    )


def _load_worker_results(spec: WorkerSpec) -> list[dict[str, Any]]:
    worker_output = json.loads(spec.output_path.read_text(encoding="utf-8"))
    if not isinstance(worker_output, dict) or worker_output.get("condition") != spec.condition:
        raise RuntimeError(f"Worker output has the wrong condition: {spec.output_path}")
    results = worker_output.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"Worker output has no result list: {spec.output_path}")
    rows = [validate_result_row(row) for row in results if isinstance(row, dict)]
    expected_user_counts = set(spec.user_counts)
    measured_user_counts = {row["num_users"] for row in rows}
    if measured_user_counts != expected_user_counts:
        raise RuntimeError(
            f"Worker output user-count mismatch for {spec.condition}: "
            f"expected={sorted(expected_user_counts)} actual={sorted(measured_user_counts)}"
        )
    return rows


def _validate_runtime_requirements(config: ThroughputConfig) -> None:
    _require_lmcache_settings(config)
    _require_runtime_modules(config)
    _require_dataset(config)
    _require_fact_catalogs(config)
    _require_embedding_access(config)


def _require_lmcache_settings(config: ThroughputConfig) -> None:
    if LMCACHE_CONDITION in config.conditions:
        lmcache_config(config)


def _require_runtime_modules(config: ThroughputConfig) -> None:
    required = set(REQUIRED_MODULES)
    for condition in config.conditions:
        required.update(CONDITION_MODULES[condition])
    missing = sorted(module for module in required if importlib.util.find_spec(module) is None)
    if missing:
        raise RuntimeError(
            "Throughput runtime modules are missing: "
            + ", ".join(missing)
            + ". Run scripts/setup_remote.sh in the GPU environment first."
        )


def _require_dataset(config: ThroughputConfig) -> None:
    if not Path(config.dataset_path).exists():
        raise RuntimeError(
            f"LoCoMo dataset not found at {config.dataset_path}; set dataset_path in the run "
            "JSON or download it with scripts/setup_remote.sh."
        )


def _require_fact_catalogs(config: ThroughputConfig) -> None:
    catalog_root = Path(config.memory_llm_cache_dir) / "fact-catalogs"
    if not any(catalog_root.rglob("*.json")):
        raise RuntimeError(
            f"No Mem0 fact catalogs found under {catalog_root}. Run the extract-facts stage "
            "with matching extraction settings first."
        )


def _require_embedding_access(config: ThroughputConfig) -> None:
    if not config.embedding_api_key and not config.embedding_base_url:
        raise RuntimeError(
            "Every throughput condition embeds queries live; set OPENAI_API_KEY or "
            "inferscale.embedding.base_url in the run JSON."
        )


def _validate_existing_config(config: ThroughputConfig) -> None:
    snapshot_path = _config_snapshot_path(config)
    if not snapshot_path.exists():
        return
    saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ConfigError(f"Existing config is not a JSON object: {snapshot_path}")
    for key, expected in config.to_jsonable().items():
        if key in VOLATILE_CONFIG_KEYS:
            continue
        if key not in saved:
            raise ConfigError(
                f"Run directory {config.run_dir} was written without the {key} setting; "
                "set a new run_id in the run JSON instead of resuming it."
            )
        if saved[key] != expected:
            raise ConfigError(
                f"Run directory {config.run_dir} already contains a different {key}. "
                "Set a new run_id in the run JSON."
            )


def _worker_environment(config: ThroughputConfig) -> dict[str, str]:
    environment = dict(os.environ)
    if config.embedding_api_key:
        environment[WORKER_API_KEY_VARIABLE] = config.embedding_api_key
    return environment
