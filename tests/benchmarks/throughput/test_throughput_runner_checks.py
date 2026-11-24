from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.throughput import runner, worker
from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.reporting import build_result_row, read_existing_results
from benchmarks.memory.throughput.runner import run_throughput


def runnable_config(tmp_path: Path, **overrides: Any) -> ThroughputConfig:
    settings: dict[str, Any] = {
        "run_id": "runner-checks",
        "conditions": ("mem0_qdrant",),
        "user_counts": (2, 3),
        "dataset_path": tmp_path / "locomo10.json",
        **overrides,
    }
    config = make_throughput_config(tmp_path, **settings)
    config.dataset_path.write_text("[]", encoding="utf-8")
    catalog = config.memory_llm_cache_dir / "fact-catalogs" / "mem0-2" / "sample.json"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("{}", encoding="utf-8")
    return config


def install_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    def find_spec(name: str) -> object:
        return object()

    monkeypatch.setattr(
        runner, "importlib", SimpleNamespace(util=SimpleNamespace(find_spec=find_spec))
    )


def measured_rows(config: ThroughputConfig, condition: str, user_counts: Any) -> list[dict]:
    return [
        build_result_row(
            config,
            num_users,
            condition=condition,
            generation_time_s=1.0,
            total_input_tokens=num_users * 20,
            total_output_tokens=num_users * 10,
        )
        for num_users in user_counts
    ]


class FakeWorkers:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, failing_job_number: int | None = None):
        self.jobs: list[dict[str, Any]] = []
        self.failing_job_number = failing_job_number
        monkeypatch.setattr(runner, "collect_system_metadata", lambda: {"platform": "test"})
        monkeypatch.setattr(runner.subprocess, "run", self.run)
        monkeypatch.setattr(worker, "configure_logging", lambda level: None)
        monkeypatch.setattr(worker, "run_condition", measured_rows)

    def run(self, command, *, input, text, check, cwd):
        job = json.loads(input)
        self.jobs.append(job)
        if len(self.jobs) == self.failing_job_number:
            raise subprocess.CalledProcessError(1, command)
        worker.main(job)


def test_rows_are_saved_after_every_worker_so_an_interrupted_run_keeps_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runnable_config(tmp_path, conditions=("mem0_qdrant", "mem0_jasper"))
    install_modules(monkeypatch)
    FakeWorkers(monkeypatch, failing_job_number=2)

    with pytest.raises(subprocess.CalledProcessError):
        run_throughput(config)

    saved = read_existing_results(config.run_dir)
    assert [(row["condition"], row["num_users"]) for row in saved] == [
        ("mem0_qdrant", 2),
        ("mem0_qdrant", 3),
    ]
    summary = json.loads((config.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 2


@pytest.mark.parametrize(
    "changed_setting",
    [
        pytest.param({"conditions": ("mem0_jasper",)}, id="conditions can be added later"),
        pytest.param(
            {"local_store_dir": Path("/another/scratch")},
            id="the scratch store path can move between hosts",
        ),
        pytest.param({"log_level": "DEBUG"}, id="the log level is operational"),
    ],
)
def test_resumed_run_may_change_a_setting_that_leaves_the_measured_workload_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_setting: dict[str, Any]
) -> None:
    install_modules(monkeypatch)
    FakeWorkers(monkeypatch)
    first_settings = {"memory_llm_base_url": "http://localhost:9000/v1"}
    run_throughput(runnable_config(tmp_path, **first_settings))

    resumed = run_throughput(runnable_config(tmp_path, **first_settings, **changed_setting))

    assert resumed["row_count"] >= 2


def test_run_announces_each_worker_and_the_number_of_rows_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, throughput_log: list[str]
) -> None:
    config = runnable_config(tmp_path, conditions=("mem0_qdrant", "kv_injection"))
    install_modules(monkeypatch)
    FakeWorkers(monkeypatch)

    run_throughput(config)

    worker_results = config.run_dir / "worker-results"
    assert throughput_log == [
        "[1/3] mem0_qdrant users=2,3",
        f"worker wrote 2 row(s) to {worker_results / 'mem0_qdrant.json'}",
        "[2/3] kv_injection users=2",
        f"worker wrote 1 row(s) to {worker_results / 'kv_injection-2u.json'}",
        "[3/3] kv_injection users=3",
        f"worker wrote 1 row(s) to {worker_results / 'kv_injection-3u.json'}",
        f"wrote 4 throughput rows to {config.run_dir}",
    ]
