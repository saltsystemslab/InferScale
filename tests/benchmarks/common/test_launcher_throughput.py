from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig
from benchmarks.common.launcher import expand_runs
from benchmarks.memory.throughput.config import ThroughputConfig


def _expand(tmp_path: Path, data: dict) -> tuple[list[dict], RuntimeConfig, Path]:
    runtime = RuntimeConfig.from_dict({}, root=tmp_path)
    path = tmp_path / "throughput.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    plan = {"family": "throughput", "configs": [str(path)], "sweep": "cpu-store"}
    return expand_runs(plan, runtime, "run", "test-stamp"), runtime, path


def test_throughput_cpu_sweep_materializes_missing_kv_section(tmp_path: Path) -> None:
    data = {
        "benchmark": "memory-throughput",
        "model": "test/model",
        "inferscale": {},
        "sweeps": {"cpu-store": {"conditions": ["kv_injection"], "kv_store_backend": "cpu"}},
    }
    original = deepcopy(data)
    runs, runtime, path = _expand(tmp_path, data)

    assert len(runs) == 1
    config = ThroughputConfig.from_dict(runs[0], runtime=runtime)
    assert config.kv_store_backend == "cpu"
    assert config.kv_staging_slots == 4
    assert config.conditions == ("kv_injection",)
    assert config.run_id == "throughput-test-stamp-model"
    assert data == original == json.loads(path.read_text(encoding="utf-8"))


def test_throughput_sweep_preserves_other_kv_settings(tmp_path: Path) -> None:
    data = {
        "benchmark": "memory-throughput",
        "model": "test/model",
        "inferscale": {"kv": {"dtype": "float16", "max_position": 8192}},
        "sweeps": {"cpu-store": {"kv_store_backend": "cpu", "kv_staging_slots": 8}},
    }
    runs, runtime, _ = _expand(tmp_path, data)
    config = ThroughputConfig.from_dict(runs[0], runtime=runtime)

    assert config.kv_store_backend == "cpu"
    assert config.kv_staging_slots == 8
    assert config.kv_dtype == "float16"
    assert config.kv_max_position == 8192


def test_throughput_launch_rejects_invalid_sweep_before_expansion(tmp_path: Path) -> None:
    data = {
        "benchmark": "memory-throughput",
        "model": "test/model",
        "inferscale": {},
        "sweeps": {"cpu-store": {"kv_store_backend": "cpu", "kv_staging_slots": True}},
    }
    with pytest.raises(ConfigError, match="kv_staging_slots"):
        _expand(tmp_path, data)
