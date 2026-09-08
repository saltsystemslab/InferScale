from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def main(
    config_path: Path,
    *,
    runtime_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    from benchmarks.common.config import load_runtime_config
    from benchmarks.memory.throughput.config import load_throughput_config
    from benchmarks.memory.throughput.runner import run_throughput

    runtime = load_runtime_config(runtime_path)
    config = load_throughput_config(config_path, runtime)
    runtime.apply_environment()
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s | %(levelname)-5s | %(message)s",
    )
    return run_throughput(config, dry_run=dry_run)
