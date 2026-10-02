"""Logging setup for the benchmark CLIs."""

from __future__ import annotations

import logging
import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Route loguru (harness) and stdlib logging (library) to stderr at one level."""
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
