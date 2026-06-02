from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

if "SCRATCH_ROOT" in os.environ:
    _DEFAULT_CACHE_ROOT = Path(os.environ["SCRATCH_ROOT"]) / "cache"
else:
    _DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache"
os.environ.setdefault("BENCHMARK_CACHE_ROOT", str(_DEFAULT_CACHE_ROOT))
os.environ.setdefault("MEM0_DIR", str(Path(os.environ["BENCHMARK_CACHE_ROOT"]) / "mem0"))

from .config import parse_args
from .runner import run_benchmark


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    _configure_logging()
    summary = run_benchmark(config)
    print(f"wrote results to {config.run_dir}")
    if summary.get("accuracy") is not None:
        print(f"accuracy={summary['accuracy']:.4f} judged={summary['judged_count']}/{summary['question_count']}")
    else:
        print(f"questions={summary['question_count']} judged={summary['judged_count']}")


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.environ.get("LOCOMO_LOG_LEVEL", "INFO").upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    )


if __name__ == "__main__":
    main()
