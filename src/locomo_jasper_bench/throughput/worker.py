"""Subprocess entry point that runs one throughput condition.

The runner invokes this module (`python -m locomo_jasper_bench.throughput.worker`)
once per condition/user-count so each measurement gets a fresh CUDA process.
Condition implementations live in kv_condition.py and mem0_condition.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..results import write_json
from .config import (
    ALL_CONDITIONS,
    ThroughputConfig,
    condition_vector_backend,
    parse_user_counts,
)
from .kv_condition import run_kv_injection
from .mem0_condition import run_mem0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Internal throughput benchmark worker.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=ALL_CONDITIONS, required=True)
    parser.add_argument("--user-counts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = ThroughputConfig.from_json_file(args.config)
    user_counts = parse_user_counts(args.user_counts)
    unknown = [count for count in user_counts if count not in config.user_counts]
    if unknown:
        parser.error(
            "Worker user counts are not in the configured list: " + ", ".join(map(str, unknown))
        )

    results = run_condition(config, args.condition, user_counts)
    write_json(args.output, {"condition": args.condition, "results": results})
    print(f"worker wrote {len(results)} row(s) to {args.output}", flush=True)


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
    main()
