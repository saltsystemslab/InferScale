from __future__ import annotations

import logging
import os
import sys

from ..runtime_paths import configure_runtime_environment

configure_runtime_environment()


def main(argv: list[str] | None = None) -> None:
    from .config import parse_args
    from .runner import run_throughput

    config, dry_run = parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOCOMO_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)-5s | %(message)s",
    )
    try:
        run_throughput(config, dry_run=dry_run)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
