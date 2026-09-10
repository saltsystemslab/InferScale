from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferscale.v1 import (
    EmbeddingConfig,
    EngineConfig,
    GenerationConfig,
    InferScaleConfig,
    JasperIndexConfig,
    KVConfig,
    PromptConfig,
)
from inferscale.v1.config import DEFAULT_CONNECTOR_MODULE


def test_defaults_match_the_serving_contract() -> None:
    config = InferScaleConfig(model="meta-llama/Llama-3.1-8B-Instruct")
    assert config.engine.connector_module == DEFAULT_CONNECTOR_MODULE == "inferscale.v1.kv.connector"
    assert config.engine.block_size == 16
    assert config.kv.store_backend == "gpu"
    assert config.kv.max_position == 32768
    assert config.index.beam_width == 64
    assert config.embedding.model == "text-embedding-3-small"
    assert config.context_window == 0
    assert config.vector_backend == "exact"
    assert config.to_dict()["context_window"] == 0
    assert InferScaleConfig.from_dict({"model": config.model}).context_window == 0


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
        ({"model": "m", "vector_backend": "jasper", "top_k": 960}, "beam width"),
        ({"model": "m", "vector_backend": "jasper", "index": {"beam_width": 1000}}, "beam width"),
        ({"model": " "}, "model must be"),
    ],
)
def test_from_dict_rejects_invalid_values(data, message) -> None:
    with pytest.raises(ValueError, match=message):
        InferScaleConfig.from_dict(data)


@pytest.mark.parametrize("vector_backend", ["exact", "jasper"])
def test_vector_backend_round_trips(tmp_path: Path, vector_backend: str) -> None:
    config = InferScaleConfig(model="m", vector_backend=vector_backend)
    data = config.to_dict()
    assert data["vector_backend"] == vector_backend
    assert InferScaleConfig.from_dict(data) == config
    path = tmp_path / "inferscale.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert InferScaleConfig.from_json(path) == config


@pytest.mark.parametrize("vector_backend", ["matmul", "cpu", "qdrant", "", True, None, [], {}])
@pytest.mark.parametrize("source", ["constructor", "dict", "json"])
def test_vector_backend_rejects_invalid_values(tmp_path, vector_backend, source) -> None:
    data = {"model": "m", "vector_backend": vector_backend}
    with pytest.raises(ValueError, match="vector_backend"):
        if source == "constructor":
            InferScaleConfig(**data)
        elif source == "dict":
            InferScaleConfig.from_dict(data)
        else:
            path = tmp_path / "inferscale.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            InferScaleConfig.from_json(path)


@pytest.mark.parametrize("selector", [{}, {"vector_backend": "exact"}])
def test_exact_has_no_jasper_beam_width_limit(selector) -> None:
    config = InferScaleConfig.from_dict({
        "model": "m",
        **selector,
        "top_k": 2000,
        "index": {"beam_width": 1000},
    })
    assert config.top_k == 2000
    assert config.index.beam_width == 1000


@pytest.mark.parametrize("context_window", [0, 1, 3, 100])
def test_context_window_round_trips_through_all_config_entrypoints(
    tmp_path: Path, context_window: int
) -> None:
    config = InferScaleConfig(model="m", context_window=context_window)
    data = config.to_dict()
    assert data["context_window"] == context_window
    assert InferScaleConfig.from_dict(data) == config

    path = tmp_path / "inferscale.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert InferScaleConfig.from_json(path) == config


@pytest.mark.parametrize("context_window", [-1, True, False, 1.0, "1", None, [], {}])
@pytest.mark.parametrize("source", ["constructor", "dict", "json"])
def test_context_window_rejects_negative_or_noninteger_values(
    tmp_path: Path, context_window, source: str
) -> None:
    data = {"model": "m", "context_window": context_window}
    path = tmp_path / "inferscale.json"
    if source == "json":
        path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="context_window"):
        if source == "constructor":
            InferScaleConfig(**data)
        elif source == "dict":
            InferScaleConfig.from_dict(data)
        else:
            InferScaleConfig.from_json(path)


def test_context_window_preserves_existing_positional_config_sections() -> None:
    engine = EngineConfig(block_size=32)
    kv = KVConfig(staging_slots=8)
    index = JasperIndexConfig(beam_width=128)
    embedding = EmbeddingConfig(batch_size=4)
    prompt = PromptConfig(chunk_separator="\n\n")
    generation = GenerationConfig(max_tokens=64)
    config = InferScaleConfig(
        "m", 5, engine, kv, index, embedding, prompt, generation,
        context_window=2, vector_backend="exact",
    )

    assert config.top_k == 5
    assert config.context_window == 2
    assert config.vector_backend == "exact"
    assert config.engine is engine
    assert config.kv is kv
    assert config.index is index
    assert config.embedding is embedding
    assert config.prompt is prompt
    assert config.generation is generation

    with pytest.raises(TypeError):
        InferScaleConfig("m", 5, engine, kv, index, embedding, prompt, generation, 2)
