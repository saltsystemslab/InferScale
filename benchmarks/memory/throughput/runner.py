"""Run every throughput condition in its own worker subprocess and merge the result rows.

Each worker gets a fresh CUDA process so conditions cannot share engine state;
kv_injection additionally gets one worker per user count because its composed
memories stay GPU-resident for the whole measurement. Rows are merged into the
run directory after every worker, so an interrupted run resumes by rerunning
the same run id.
"""

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
from benchmarks.common.paths import project_root
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

# Settings that may differ between invocations of the same run id without
# changing the measured workload: conditions can be added later, the key is
# redacted in the snapshot, the scratch store path can move between hosts, and
# the log level is operational.
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
                WorkerSpec(condition, (count,), worker_dir / f"{condition}-{count}u.json")
                for count in config.user_counts
            )
        else:
            specs.append(
                WorkerSpec(condition, config.user_counts, worker_dir / f"{condition}.json")
            )
    return specs


def build_worker_job(config: ThroughputConfig, spec: WorkerSpec) -> dict[str, Any]:
    return {
        "config_path": str((config.run_dir / "config.json").resolve()),
        "condition": spec.condition,
        "user_counts": list(spec.user_counts),
        "output_path": str(spec.output_path.resolve()),
    }


def run_throughput(config: ThroughputConfig, *, dry_run: bool = False) -> dict[str, Any] | None:
    specs = worker_specs(config)
    if dry_run:
        print(f"run directory: {config.run_dir}")
        print(f"model: {config.model}")
        print(f"user counts: {user_counts_text(config.user_counts)}")
        for spec in specs:
            print(json.dumps(build_worker_job(config, spec)))
        return None

    _validate_runtime_requirements(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_config(config)
    rows = read_existing_results(config.run_dir)
    write_json(config.run_dir / "config.json", config.to_jsonable())
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / "system.json", system_metadata)

    for index, spec in enumerate(specs, start=1):
        logger.info(
            "[{}/{}] {} users={}",
            index,
            len(specs),
            spec.condition,
            user_counts_text(spec.user_counts),
        )
        _run_worker(config, spec)
        rows = merge_result_rows(rows, _load_worker_results(spec))
        write_reports(config, rows, system_metadata=system_metadata)

    summary = write_reports(config, rows, system_metadata=system_metadata)
    print(f"wrote {summary['row_count']} throughput rows to {config.run_dir}")
    return summary


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
    raw = json.loads(spec.output_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("condition") != spec.condition:
        raise RuntimeError(f"Worker output has the wrong condition: {spec.output_path}")
    raw_results = raw.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError(f"Worker output has no result list: {spec.output_path}")
    rows = [validate_result_row(row) for row in raw_results if isinstance(row, dict)]
    expected = set(spec.user_counts)
    actual = {row["num_users"] for row in rows}
    if actual != expected:
        raise RuntimeError(
            f"Worker output user-count mismatch for {spec.condition}: "
            f"expected={sorted(expected)} actual={sorted(actual)}"
        )
    return rows


def _validate_runtime_requirements(config: ThroughputConfig) -> None:
    if LMCACHE_CONDITION in config.conditions:
        lmcache_config(config)
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
    if not Path(config.dataset_path).exists():
        raise RuntimeError(
            f"LoCoMo dataset not found at {config.dataset_path}; set dataset_path in the run "
            "JSON or download it with scripts/setup_remote.sh."
        )
    catalog_root = Path(config.memory_llm_cache_dir) / "fact-catalogs"
    if not any(catalog_root.rglob("*.json")):
        raise RuntimeError(
            f"No Mem0 fact catalogs found under {catalog_root}. Run the extract-facts stage "
            "with matching extraction settings first."
        )
    if not config.embedding_api_key and not config.embedding_base_url:
        raise RuntimeError(
            "Every throughput condition embeds queries live; set OPENAI_API_KEY or "
            "inferscale.embedding.base_url in the run JSON."
        )


def _validate_existing_config(config: ThroughputConfig) -> None:
    """A resumed run id must carry exactly the settings its first invocation recorded."""
    path = config.run_dir / "config.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ConfigError(f"Existing config is not a JSON object: {path}")
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
