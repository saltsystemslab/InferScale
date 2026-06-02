from __future__ import annotations

import os
import sys

from loguru import logger

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
