from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.files import write_json
from benchmarks.common.json_fields import ConfigError, int_list, reject_unknown_keys, require
from benchmarks.common.logging import configure_logging
from benchmarks.memory.throughput.config import ThroughputConfig, condition_vector_backend
from benchmarks.memory.throughput.kv_condition import run_kv_injection
from benchmarks.memory.throughput.lmcache_condition import run_lmcache_blend
from benchmarks.memory.throughput.mem0_condition import run_mem0

JOB_KEYS = ("config_path", "condition", "user_counts", "output_path")
WHERE = "throughput worker job"


def main(job: dict[str, Any]) -> None:
    if not isinstance(job, dict):
        raise ConfigError(f"{WHERE} must be a JSON object.")
    reject_unknown_keys(job, JOB_KEYS, WHERE)
    config_path = Path(require(job, "config_path", str, WHERE))
    condition = require(job, "condition", str, WHERE)
    output_path = Path(require(job, "output_path", str, WHERE))
    user_counts = int_list(job, "user_counts", WHERE)
    config = ThroughputConfig.from_json_file(config_path)
    if condition not in config.conditions:
        raise ConfigError(f"Worker condition {condition!r} is not in the configured list.")
    unconfigured_counts = [count for count in user_counts if count not in config.user_counts]
    if unconfigured_counts:
        raise ConfigError(
            "Worker user counts are not in the configured list: "
            + ", ".join(map(str, unconfigured_counts))
        )
    configure_logging(config.log_level)
    results = run_condition(config, condition, user_counts)
    write_json(output_path, {"condition": condition, "results": results})
    logger.info("worker wrote {} row(s) to {}", len(results), output_path)


def run_condition(
    config: ThroughputConfig, condition: str, user_counts: tuple[int, ...]
) -> list[dict[str, Any]]:
    if condition == "kv_injection":
        if len(user_counts) != 1:
            raise ValueError("A KV worker must receive exactly one user count.")
        return [run_kv_injection(config, user_counts[0])]
    if condition in {"mem0_qdrant", "mem0_jasper"}:
        backend = condition_vector_backend(condition)
        return run_mem0(config, user_counts, condition=condition, backend=backend)
    if condition == "lmcache_blend":
        return run_lmcache_blend(config, user_counts)
    raise ValueError(f"Unsupported condition: {condition}")


if __name__ == "__main__":
    main(json.load(sys.stdin))
