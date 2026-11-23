from __future__ import annotations

import sys
from typing import Any

import pytest

from benchmarks.ttft_race import run as race


def run_command_line(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run", *arguments])
    race.main()


@pytest.fixture
def measured_sides(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    measured: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        race,
        "measure_side",
        lambda mode, *, config_path, questions, out: measured.append(
            (mode, config_path, questions, out)
        ),
    )
    monkeypatch.setattr(race, "load_timelines", lambda _: pytest.fail("measure drew a GIF"))
    return measured


def test_measure_leaves_logging_as_it_is(
    monkeypatch: pytest.MonkeyPatch,
    measured_sides: list[tuple[Any, ...]],
    logging_configurations: list[str],
) -> None:
    run_command_line(monkeypatch, "measure", "--mode", "inferscale", "--out", "inferscale.json")

    assert len(measured_sides) == 1
    assert logging_configurations == []
