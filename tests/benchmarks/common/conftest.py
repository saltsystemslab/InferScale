from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

from benchmarks.common.config import RuntimeConfig, load_json_object
from benchmarks.common.launcher import execute

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runtime(tmp_path: Path) -> RuntimeConfig:
    data = load_json_object(ROOT / "configs/runtime.json")
    data["storage"]["runtime_root"] = str(tmp_path)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(data))
    return RuntimeConfig.from_dict(data, root=ROOT, path=path)


@pytest.fixture
def launcher_log(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    monkeypatch.setattr(execute, "configure_logging", lambda: None)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink)
