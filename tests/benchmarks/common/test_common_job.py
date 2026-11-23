from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.common.job import execute

RUN = {"config": "run.json", "runtime_config": "runtime.json"}


@pytest.fixture
def started_benchmarks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path, dict[str, Any]]]:
    started: list[tuple[str, Path, dict[str, Any]]] = []

    def entry_point(benchmark: str):
        def main(config_path: Path, **options: Any) -> None:
            started.append((benchmark, config_path, options))

        return main

    monkeypatch.setattr("benchmarks.memory.run.main", entry_point("memory"))
    monkeypatch.setattr("benchmarks.rag.run.main", entry_point("rag"))
    monkeypatch.setattr("benchmarks.memory.throughput.run.main", entry_point("throughput"))
    return started


@pytest.mark.parametrize("family", ["memory", "rag"])
@pytest.mark.parametrize(
    "authored_stage,stage",
    [({"stage": "judge"}, "judge")],
)
def test_memory_and_rag_jobs_start_their_benchmark_at_the_requested_stage(
    started_benchmarks, family, authored_stage, stage
):
    execute({"family": family, **RUN, **authored_stage, "dry_run": False})

    assert started_benchmarks == [
        (family, Path("run.json"), {"runtime_path": Path("runtime.json"), "stage": stage})
    ]


@pytest.mark.parametrize("dry_run", [True, False])
def test_throughput_job_passes_the_dry_run_flag_to_its_benchmark(started_benchmarks, dry_run):
    execute({"family": "throughput", **RUN, "stage": "run", "dry_run": dry_run})

    assert started_benchmarks == [
        ("throughput", Path("run.json"), {"runtime_path": Path("runtime.json"), "dry_run": dry_run})
    ]
