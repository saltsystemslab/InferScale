"""Subprocess entry point that runs one throughput condition.

The runner invokes this module (`python -m benchmarks.memory.throughput.worker`)
once per condition/user-count so each measurement gets a fresh CUDA process.
Condition implementations live in kv_condition.py and mem0_condition.py.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from benchmarks.common.config import ConfigError, int_list, reject_unknown_keys, require
from benchmarks.common.files import write_json
from benchmarks.memory.throughput.config import (
    ThroughputConfig,
    condition_vector_backend,
)
from benchmarks.memory.throughput.kv_condition import run_kv_injection
from benchmarks.memory.throughput.mem0_condition import run_mem0


def main(job: dict[str, Any]) -> None:
    """Run one isolated measurement from the runner's JSON job."""
    where = "throughput worker job"
    if not isinstance(job, dict):
        raise ConfigError(f"{where} must be a JSON object.")
    reject_unknown_keys(job, ("config_path", "condition", "user_counts", "output_path"), where)
    config_path = Path(require(job, "config_path", str, where))
    condition = require(job, "condition", str, where)
    output_path = Path(require(job, "output_path", str, where))
    user_counts = int_list(job, "user_counts", where)
    config = ThroughputConfig.from_json_file(config_path)
    if condition not in config.conditions:
        raise ConfigError(f"Worker condition {condition!r} is not in the configured list.")
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s | %(levelname)-5s | %(message)s",
    )
    unknown = [count for count in user_counts if count not in config.user_counts]
    if unknown:
        raise ConfigError(
            "Worker user counts are not in the configured list: " + ", ".join(map(str, unknown))
        )

    results = run_condition(config, condition, user_counts)
    write_json(output_path, {"condition": condition, "results": results})
    print(f"worker wrote {len(results)} row(s) to {output_path}", flush=True)


def run_condition(
    config: ThroughputConfig,
    condition: str,
    user_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    if condition == "kv_injection":
        if len(user_counts) != 1:
            raise ValueError("A KV worker must receive exactly one user count.")
        return [run_kv_injection(config, user_counts[0])]
    if condition in {"mem0_qdrant", "mem0_jasper"}:
        backend = condition_vector_backend(condition)
        if backend is None:
            raise ValueError(f"Condition {condition} has no vector backend.")
        return run_mem0(config, user_counts, condition=condition, backend=backend)
    raise ValueError(f"Unsupported condition: {condition}")


if __name__ == "__main__":
    main(json.load(sys.stdin))
