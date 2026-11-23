from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.common.json_fields import ConfigError, optional, reject_unknown_keys, require

WHERE = "job"
JOB_KEYS = ("family", "config", "runtime_config", "stage", "dry_run")
DEFAULT_STAGE = "run"


def execute(job: dict[str, Any]) -> None:
    if not isinstance(job, dict):
        raise ConfigError("A benchmark job must contain a JSON object.")
    reject_unknown_keys(job, JOB_KEYS, WHERE)
    family = require(job, "family", str, WHERE)
    config_path = Path(require(job, "config", str, WHERE))
    runtime_path = Path(require(job, "runtime_config", str, WHERE))
    stage = optional(job, "stage", str, DEFAULT_STAGE, WHERE)
    dry_run = optional(job, "dry_run", bool, False, WHERE)
    if family == "throughput":
        _run_throughput(config_path, runtime_path, stage, dry_run=dry_run)
    elif family in {"memory", "rag"}:
        _run_memory_or_rag(family, config_path, runtime_path, stage, dry_run=dry_run)
    else:
        raise ConfigError(f"Unknown benchmark job family: {family!r}.")


def _run_throughput(config_path: Path, runtime_path: Path, stage: str, *, dry_run: bool) -> None:
    if stage != "run":
        raise ConfigError(f"Unsupported throughput stage: {stage!r}.")
    from benchmarks.memory.throughput.run import main

    main(config_path, runtime_path=runtime_path, dry_run=dry_run)


def _run_memory_or_rag(
    family: str, config_path: Path, runtime_path: Path, stage: str, *, dry_run: bool
) -> None:
    if dry_run:
        raise ConfigError("Memory and RAG dry runs must be expanded by the launcher.")
    if family == "memory":
        from benchmarks.memory.run import main
    else:
        from benchmarks.rag.run import main

    main(config_path, runtime_path=runtime_path, stage=stage)


if __name__ == "__main__":
    execute(json.load(sys.stdin))
