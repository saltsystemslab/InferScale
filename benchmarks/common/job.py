"""Internal process boundary for benchmark jobs supplied as JSON on stdin."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from benchmarks.common.config import ConfigError, optional, reject_unknown_keys, require


def execute(job: dict[str, Any]) -> None:
    if not isinstance(job, dict):
        raise ConfigError("A benchmark job must contain a JSON object.")
    reject_unknown_keys(job, ("family", "config", "runtime_config", "stage", "dry_run"), "job")
    family = require(job, "family", str, "job")
    config_path = Path(require(job, "config", str, "job"))
    runtime_path = Path(require(job, "runtime_config", str, "job"))
    stage = optional(job, "stage", str, "run", "job")
    dry_run = optional(job, "dry_run", bool, False, "job")
    if family == "throughput":
        if stage != "run":
            raise ConfigError(f"Unsupported throughput stage: {stage!r}.")
        from benchmarks.memory.throughput.run import main

        main(config_path, runtime_path=runtime_path, dry_run=dry_run)
    elif family in {"memory", "rag"}:
        if dry_run:
            raise ConfigError("Memory and RAG dry runs must be expanded by the launcher.")
        if family == "memory":
            from benchmarks.memory.run import main
        else:
            from benchmarks.rag.run import main

        main(config_path, runtime_path=runtime_path, stage=stage)
    else:
        raise ConfigError(f"Unknown benchmark job family: {family!r}.")


if __name__ == "__main__":
    execute(json.load(sys.stdin))
