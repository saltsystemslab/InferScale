from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..kv.connector_utils import DEFAULT_KV_STAGING_SLOTS, DEFAULT_KV_STORE_BACKEND
from ..results import write_json
from ..runtime_paths import project_root
from ..system import collect_system_metadata
from .config import ALL_CONDITIONS, ThroughputConfig, user_counts_text
from .reporting import (
    merge_result_rows,
    read_existing_results,
    validate_result_row,
    write_reports,
)


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
            for count in config.user_counts:
                specs.append(
                    WorkerSpec(
                        condition=condition,
                        user_counts=(count,),
                        output_path=worker_dir / f"{condition}-{count}u.json",
                    )
                )
        else:
            specs.append(
                WorkerSpec(
                    condition=condition,
                    user_counts=config.user_counts,
                    output_path=worker_dir / f"{condition}.json",
                )
            )
    return specs


def build_worker_command(config: ThroughputConfig, spec: WorkerSpec) -> list[str]:
    return [
        sys.executable,
        "-m",
        "locomo_jasper_bench.throughput.worker",
        "--config",
        str((config.run_dir / "config.json").resolve()),
        "--condition",
        spec.condition,
        "--user-counts",
        user_counts_text(spec.user_counts),
        "--output",
        str(spec.output_path.resolve()),
    ]


def run_throughput(config: ThroughputConfig, *, dry_run: bool = False) -> dict[str, Any] | None:
    specs = worker_specs(config)
    if dry_run:
        print(f"run directory: {config.run_dir}")
        print(f"model: {config.model}")
        print(f"user counts: {user_counts_text(config.user_counts)}")
        for spec in specs:
            print(shlex.join(build_worker_command(config, spec)))
        return None

    _validate_runtime_requirements(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_config(config)
    rows = read_existing_results(config.run_dir)
    completed_conditions = {str(row["condition"]) for row in rows}
    requested_conditions = set(config.conditions)
    config.conditions = tuple(
        condition
        for condition in ALL_CONDITIONS
        if condition in completed_conditions or condition in requested_conditions
    )
    write_json(config.run_dir / "config.json", config.to_jsonable())
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / "system.json", system_metadata)

    for index, spec in enumerate(specs, start=1):
        command = build_worker_command(config, spec)
        print(
            f"[{index}/{len(specs)}] {spec.condition} users={user_counts_text(spec.user_counts)}",
            flush=True,
        )
        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            command,
            check=True,
            cwd=project_root(),
            env=_worker_environment(config),
        )
        replacements = _load_worker_results(spec)
        rows = merge_result_rows(rows, replacements)
        write_reports(config, rows, system_metadata=system_metadata)

    summary = write_reports(config, rows, system_metadata=system_metadata)
    print(f"wrote {summary['row_count']} throughput rows to {config.run_dir}")
    return summary


def _load_worker_results(spec: WorkerSpec) -> list[dict[str, Any]]:
    raw = json.loads(spec.output_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("condition") != spec.condition:
        raise RuntimeError(f"Worker output has the wrong condition: {spec.output_path}")
    raw_results = raw.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError(f"Worker output has no result list: {spec.output_path}")
    rows = [validate_result_row(row) for row in raw_results if isinstance(row, dict)]
    expected = set(spec.user_counts)
    actual = {int(row["num_users"]) for row in rows}
    if actual != expected:
        raise RuntimeError(
            f"Worker output user-count mismatch for {spec.condition}: "
            f"expected={sorted(expected)} actual={sorted(actual)}"
        )
    return rows


def _validate_runtime_requirements(config: ThroughputConfig) -> None:
    retrieval_conditions = {"mem0_qdrant", "mem0_jasper", "kv_injection"}
    required_modules = {"vllm"}
    if "kv_injection" in config.conditions:
        required_modules.update(("torch", "transformers", "jasper"))
    if "mem0_qdrant" in config.conditions:
        required_modules.add("qdrant_client")
    if "mem0_jasper" in config.conditions:
        required_modules.add("jasper")
    if retrieval_conditions & set(config.conditions):
        required_modules.add("mem0")
    missing = sorted(module for module in required_modules if importlib.util.find_spec(module) is None)
    if missing:
        raise RuntimeError(
            "Throughput runtime modules are missing: "
            + ", ".join(missing)
            + ". Run scripts/setup_remote.sh in the GPU environment first."
        )
    if not Path(config.dataset_path).exists():
        raise RuntimeError(
            f"LoCoMo dataset not found at {config.dataset_path}; pass --dataset or download it "
            "with scripts/setup_remote.sh."
        )
    if retrieval_conditions & set(config.conditions):
        catalog_root = Path(config.memory_llm_cache_dir) / "fact-catalogs"
        if not any(catalog_root.rglob("*.json")):
            raise RuntimeError(
                f"No Mem0 fact catalogs found under {catalog_root}. Run "
                "locomo-jasper-bench --preembed-only with matching extraction settings first."
            )
    needs_embeddings = sorted(retrieval_conditions & set(config.conditions))
    if needs_embeddings and not config.embedding_api_key and not config.embedding_base_url:
        raise RuntimeError(
            "Conditions "
            + ", ".join(needs_embeddings)
            + " require --embedding-api-key/OPENAI_API_KEY or --embedding-base-url/OPENAI_BASE_URL."
        )


# Values a config.json written before these fields existed is treated as
# having used, so old run dirs stay resumable. Pre-change behavior was prefix
# caching hard-off and the GPU store.
_LEGACY_CONFIG_DEFAULTS: dict[str, Any] = {
    "kv_enable_prefix_caching": False,
    "kv_store_backend": DEFAULT_KV_STORE_BACKEND,
    "kv_staging_slots": DEFAULT_KV_STAGING_SLOTS,
}


def _validate_existing_config(config: ThroughputConfig) -> None:
    path = config.run_dir / "config.json"
    if not path.exists():
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Existing config is not a JSON object: {path}")
    expected_config = config.to_jsonable()
    for key, expected in expected_config.items():
        if key in {"conditions", "embedding_api_key"}:
            continue
        if key in _LEGACY_CONFIG_DEFAULTS and key not in raw:
            recorded = _LEGACY_CONFIG_DEFAULTS[key]
        else:
            recorded = raw.get(key)
        if recorded != expected:
            raise RuntimeError(
                f"Run directory {config.run_dir} already contains a different {key}. Use a new --run-id."
            )


def _worker_environment(config: ThroughputConfig) -> dict[str, str]:
    environment = dict(os.environ)
    if config.embedding_api_key:
        environment["LOCOMO_THROUGHPUT_EMBEDDING_API_KEY"] = config.embedding_api_key
    return environment
