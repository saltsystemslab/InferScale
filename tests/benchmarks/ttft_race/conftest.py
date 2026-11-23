from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger

from benchmarks.ttft_race import run as race


@pytest.fixture(autouse=True)
def logging_configurations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    configurations: list[str] = []
    monkeypatch.setattr(race, "configure_logging", lambda: configurations.append("configured"))
    return configurations


@pytest.fixture
def race_log() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink)
