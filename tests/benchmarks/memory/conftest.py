from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
from loguru import logger

from benchmarks.memory import run


@pytest.fixture
def memory_log() -> Iterator[list[tuple[str, str]]]:
    logged: list[tuple[str, str]] = []
    sink = logger.add(
        lambda message: logged.append((message.record["level"].name, message.record["message"])),
        level="DEBUG",
        format="{message}",
    )
    yield logged
    with contextlib.suppress(ValueError):
        logger.remove(sink)


@pytest.fixture
def stage_log(
    monkeypatch: pytest.MonkeyPatch, memory_log: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    monkeypatch.setattr(run, "configure_logging", lambda level: None)
    return memory_log
