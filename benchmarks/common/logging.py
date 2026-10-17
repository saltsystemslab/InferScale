"""Logging setup for the benchmark entry points.

The benchmarks log through loguru while the ``inferscale`` library and its
dependencies use the standard library, so both are routed to stderr at the
same level and in the same format.
"""

from __future__ import annotations

import logging
import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    resolved = level.upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=resolved,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    )
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        stream=sys.stderr,
        force=True,
    )
