from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import pytest

from benchmarks.common import serve
from benchmarks.common.config import RuntimeConfig

JUDGE_SERVER = {"base_url": "http://127.0.0.1:8123/v1", "model": "judge/model"}
JUDGE_SERVER_COMMAND = [
    "vllm",
    "serve",
    "judge/model",
    "--host",
    "127.0.0.1",
    "--port",
    "8123",
    "--trust-remote-code",
    "--dtype",
    "auto",
    "--max-model-len",
    "8192",
    "--gpu-memory-utilization",
    "0.8",
    "--tensor-parallel-size",
    "1",
]
SERVER_EXIT_CODE = 3


def write_serve_config(folder: Path, judge_server: dict[str, Any], **settings: Any) -> Path:
    runtime_path = folder / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "storage": {"runtime_root": str(folder / "data")},
                "judge_server": judge_server,
            }
        )
    )
    serve_config_path = folder / "serve.json"
    serve_config_path.write_text(json.dumps({"runtime_config": str(runtime_path), **settings}))
    return serve_config_path


class ServeWorkflow:
    def __init__(
        self, folder: Path, monkeypatch: pytest.MonkeyPatch, announcements: list[str]
    ) -> None:
        self.folder = folder
        self.runtime_root = folder / "data"
        self.announcements = announcements
        self.events: list[Any] = []
        monkeypatch.delenv("JUDGE_LLM_API_KEY", raising=False)
        monkeypatch.setattr(serve, "configure_logging", lambda: self.events.append("logging"))
        monkeypatch.setattr(
            RuntimeConfig, "apply_environment", lambda runtime: self.events.append("environment")
        )
        monkeypatch.setattr(serve.subprocess, "call", self.start_server)

    def start_server(self, command: list[str]) -> int:
        self.events.append(list(command))
        return SERVER_EXIT_CODE

    def run(self, judge_server: dict[str, Any], **settings: Any) -> int:
        return serve.serve(write_serve_config(self.folder, judge_server, **settings))


@pytest.fixture
def workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_log: list[str]
) -> ServeWorkflow:
    return ServeWorkflow(tmp_path, monkeypatch, announcements=serve_log)


def test_serve_starts_the_judge_server_and_returns_its_exit_code(workflow):
    assert workflow.run(JUDGE_SERVER) == SERVER_EXIT_CODE

    assert workflow.events == [
        "logging",
        "environment",
        [*JUDGE_SERVER_COMMAND, "--api-key", "token-abc123"],
    ]
    assert (workflow.runtime_root / "results").is_dir()
    assert workflow.announcements == [shlex.join(JUDGE_SERVER_COMMAND)]
