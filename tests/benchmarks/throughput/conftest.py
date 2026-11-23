from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from loguru import logger

from benchmarks.memory.throughput import worker


@pytest.fixture
def throughput_log(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    monkeypatch.setattr(worker, "configure_logging", lambda level: None)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink)


@pytest.fixture
def ticking_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[str]], None]:
    def install(events: list[str]) -> None:
        readings = iter(range(1, 10_000))

        def perf_counter() -> float:
            events.append("clock")
            return float(next(readings) ** 2)

        monkeypatch.setattr(time, "perf_counter", perf_counter)

    return install
