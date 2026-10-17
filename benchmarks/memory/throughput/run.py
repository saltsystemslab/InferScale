"""Entry point of the throughput benchmark, invoked by the launcher job."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def main(
    config_path: Path, *, runtime_path: Path | None = None, dry_run: bool = False
) -> dict[str, Any] | None:
    from benchmarks.common.config import load_runtime_config
    from benchmarks.common.logging import configure_logging
    from benchmarks.memory.throughput.config import load_throughput_config
    from benchmarks.memory.throughput.runner import run_throughput

    runtime = load_runtime_config(runtime_path)
    config = load_throughput_config(config_path, runtime)
    runtime.apply_environment()
    configure_logging(config.log_level)
    return run_throughput(config, dry_run=dry_run)
