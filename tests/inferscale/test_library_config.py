from __future__ import annotations

import json

import pytest

from inferscale.v1 import InferScaleConfig
from inferscale.v1.config import DEFAULT_CONNECTOR_MODULE


def test_defaults_match_the_serving_contract() -> None:
    config = InferScaleConfig(model="meta-llama/Llama-3.1-8B-Instruct")
    assert config.engine.connector_module == DEFAULT_CONNECTOR_MODULE == "inferscale.v1.kv.connector"
    assert config.engine.block_size == 16
    assert config.kv.store_backend == "gpu"
    assert config.kv.max_position == 32768
    assert config.index.beam_width == 64
    assert config.embedding.model == "text-embedding-3-small"


def test_from_dict_builds_nested_sections_and_round_trips(tmp_path) -> None:
    data = {
        "model": "m",
        "top_k": 5,
        "engine": {"gpu_memory_utilization": 0.5, "block_size": 32},
        "kv": {"store_backend": "cpu", "staging_slots": 9},
        "index": {"beam_width": 128},
        "embedding": {"batch_size": 4},
        "generation": {"max_tokens": 64},
    }
    config = InferScaleConfig.from_dict(data)
    assert config.engine.block_size == 32
    assert config.engine.max_model_len == 32768
    assert config.kv.store_backend == "cpu"
    assert config.kv.staging_slots == 9
    assert config.index.beam_width == 128
    assert config.generation.max_tokens == 64
    assert InferScaleConfig.from_dict(config.to_dict()) == config

    path = tmp_path / "inferscale.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert InferScaleConfig.from_json(path) == config


@pytest.mark.parametrize(
    "data, message",
    [
        ({"model": "m", "unknown": 1}, "unknown keys: unknown"),
        ({"model": "m", "kv": {"backend": "gpu"}}, "inferscale.kv has unknown keys: backend"),
        ({"model": "m", "top_k": 0}, "top_k must be >= 1"),
        ({"model": "m", "kv": {"store_backend": "disk"}}, "kv.store_backend"),
        ({"model": "m", "kv": {"dtype": "int8"}}, "kv.dtype"),
        ({"model": "m", "engine": {"gpu_memory_utilization": 1.0}}, "gpu_memory_utilization"),
        ({"model": "m", "top_k": 960}, "beam width"),
        ({"model": "m", "index": {"beam_width": 1000}}, "beam width"),
        ({"model": " "}, "model must be"),
    ],
)
def test_from_dict_rejects_invalid_values(data, message) -> None:
    with pytest.raises(ValueError, match=message):
        InferScaleConfig.from_dict(data)
