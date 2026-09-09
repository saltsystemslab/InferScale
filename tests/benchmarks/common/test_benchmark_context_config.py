from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.rag.config import RagBenchConfig


@pytest.fixture(params=[
    (MemoryRunConfig, "configs/memory/accuracy-latency/llama.json", 0),
    (ThroughputConfig, "configs/memory/throughput/llama.json", 50),
    (RagBenchConfig, "configs/rag/multihoprag.json", 5),
], ids=["memory", "throughput", "rag"])
def benchmark_input(request, tmp_path):
    config_type, relative_path, default_window = request.param
    root = Path(__file__).resolve().parents[3]
    data = json.loads((root / relative_path).read_text(encoding="utf-8"))
    runtime = RuntimeConfig.from_dict({}, root=tmp_path)
    return config_type, data, runtime, default_window


@pytest.mark.parametrize("context_window", [None, 0, 3])
def test_context_window_is_consistent_with_the_library_config(
    benchmark_input, context_window
) -> None:
    config_type, data, runtime, default_window = benchmark_input
    if context_window is None:
        data.pop("context_window", None)
        expected = default_window
    else:
        data["context_window"] = context_window
        expected = context_window

    config = config_type.from_dict(data, runtime=runtime)
    assert config.context_window == expected
    if isinstance(config, MemoryRunConfig):
        assert config.inferscale.context_window == expected
        assert config.to_jsonable()["inferscale"]["context_window"] == expected
    assert "context_window" not in data["inferscale"]


@pytest.mark.parametrize("nested_context_window", [0, 3, None])
def test_context_window_has_one_authored_source(benchmark_input, nested_context_window) -> None:
    config_type, data, runtime, _ = benchmark_input
    data["context_window"] = 3
    data["inferscale"]["context_window"] = nested_context_window
    with pytest.raises(ConfigError, match=r"inferscale\.context_window is set from the top level"):
        config_type.from_dict(data, runtime=runtime)
