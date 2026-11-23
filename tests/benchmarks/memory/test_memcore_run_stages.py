from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from memory_config import PROJECT_ROOT, memory_config_data

from benchmarks.memory import run, runner

JUDGED_SUMMARY = {"metrics": {"accuracy": 2 / 3}, "judged_count": 2, "question_count": 3}
UNJUDGED_SUMMARY = {"metrics": {"accuracy": None}, "judged_count": 0, "question_count": 3}

LaunchStage = Callable[..., None]


@pytest.fixture
def launch_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_log) -> LaunchStage:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"storage": {"runtime_root": str(tmp_path)}}))

    def launch(stage: str, **settings: Any) -> None:
        config_path = tmp_path / "memory.json"
        config_path.write_text(json.dumps(memory_config_data(run_id="stage-run", **settings)))
        run.main(config_path, runtime_path=runtime_path, stage=stage)

    return launch


@pytest.mark.parametrize(
    "summary,accuracy_line",
    [
        (JUDGED_SUMMARY, "accuracy=0.6667 judged=2/3"),
        (UNJUDGED_SUMMARY, "questions=3 judged=0/3"),
    ],
)
def test_run_stage_reports_the_results_folder_and_the_accuracy(
    launch_stage, monkeypatch, stage_log, tmp_path, summary, accuracy_line
):
    monkeypatch.setattr(runner, "run_benchmark", lambda config: summary)

    launch_stage("run")

    assert stage_log == [
        ("INFO", f"wrote results to {tmp_path / 'results' / 'stage-run'}"),
        ("INFO", accuracy_line),
    ]


def test_mem0_directory_of_the_runtime_is_in_the_environment_when_the_stage_starts(
    launch_stage, monkeypatch, tmp_path
):
    mem0_dir_seen_by_stage = []

    def run_benchmark(config):
        mem0_dir_seen_by_stage.append(os.environ.get("MEM0_DIR"))
        return JUDGED_SUMMARY

    monkeypatch.setattr(runner, "run_benchmark", run_benchmark)
    os.environ.pop("MEM0_DIR", None)

    launch_stage("run")

    assert mem0_dir_seen_by_stage == [str(tmp_path / ".cache" / "mem0")]


def test_mem0_is_not_imported_until_a_stage_runs_with_the_runtime_environment_applied(tmp_path):
    imported = subprocess.run(
        [sys.executable, "-c", "import sys, benchmarks.memory.run; print('mem0' in sys.modules)"],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]),
            "MEM0_TELEMETRY": "false",
            "MEM0_DIR": str(tmp_path / "mem0"),
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert imported.stdout.strip() == "False"
