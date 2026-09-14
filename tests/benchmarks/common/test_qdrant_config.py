from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig
from benchmarks.common.qdrant_config import QdrantConfig
from benchmarks.common.vector_types import VectorStoreConfig
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.mem0.adapter import Mem0JasperVectorStore
from benchmarks.memory.mem0.memory_builder import _store_config
from benchmarks.memory.mem0.provider import build_mem0_config, register_mem0_jasper_provider
from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.stores import _vector_config


@pytest.fixture
def runtime(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig.from_dict(
        {
            "qdrant": {
                "url": "https://qdrant.example:7443",
                "grpc_port": 7444,
                "prefer_grpc": False,
                "timeout": 17.5,
                "exact": False,
            },
        },
        root=tmp_path,
    )


def test_default_qdrant_connection_uses_local_server_and_exact_search(tmp_path: Path) -> None:
    settings = RuntimeConfig.from_dict({}, root=tmp_path).qdrant

    assert settings.url == "http://127.0.0.1:6333"
    assert settings.prefer_grpc is True
    assert settings.exact is True


@pytest.mark.parametrize("settings", [
    {"url": ":memory:"},
    {"url": "file:///tmp/qdrant"},
    {"url": "http://user:secret@qdrant.example"},
    {"url": "http://qdrant.example?api_key=secret"},
    {"url": "http://qdrant.example:65536"},
    {"grpc_port": 0},
    {"grpc_port": 65536},
    {"grpc_port": True},
    {"prefer_grpc": "false"},
    {"exact": "false"},
    {"timeout": 0},
    {"timeout": -1},
    {"timeout": float("inf")},
    {"timeout": float("nan")},
    {"timeout": True},
    {"unknown_option": True},
])
def test_runtime_rejects_invalid_qdrant_settings(tmp_path: Path, settings: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        RuntimeConfig.from_dict({"qdrant": settings}, root=tmp_path)


def test_runtime_settings_reach_both_benchmarks_and_their_vector_stores(runtime: RuntimeConfig) -> None:
    data = {"model": "test/model", "inferscale": {}}
    memory = MemoryRunConfig.from_dict(data, runtime=runtime)
    throughput = ThroughputConfig.from_dict(data, runtime=runtime)

    for config in (memory, throughput):
        assert config.qdrant == runtime.qdrant
        assert config.to_jsonable()["qdrant"] == runtime.qdrant.to_dict()
    for backend in ("jasper", "qdrant"):
        assert _store_config(memory, backend=backend).qdrant == runtime.qdrant
        assert _vector_config(throughput, backend).qdrant == runtime.qdrant


def test_worker_snapshot_preserves_custom_qdrant_connection_and_search(
    tmp_path: Path, runtime: RuntimeConfig,
) -> None:
    config = ThroughputConfig.from_dict({"model": "test/model", "inferscale": {}}, runtime=runtime)
    snapshot = tmp_path / "worker.json"
    snapshot.write_text(json.dumps(config.to_jsonable()), encoding="utf-8")

    restored = ThroughputConfig.from_json_file(snapshot)

    assert restored.qdrant == runtime.qdrant
    assert isinstance(restored.qdrant, QdrantConfig)
    assert _vector_config(restored, "qdrant").qdrant == runtime.qdrant


def test_mem0_registered_schema_preserves_qdrant_settings_into_adapter(
    tmp_path: Path, runtime: RuntimeConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mem0.vector_stores.configs import VectorStoreConfig as Mem0VectorStoreConfig

    class StubStore:
        def __init__(self, root: Path, config: VectorStoreConfig) -> None:
            self.config = config

    monkeypatch.setattr("benchmarks.memory.mem0.adapter.QdrantVectorStore", StubStore)
    register_mem0_jasper_provider()
    config = build_mem0_config(
        store_root=tmp_path,
        vector_config=VectorStoreConfig(backend="qdrant", qdrant=runtime.qdrant),
        embedding_model="test-embedding",
        embedding_api_key=None,
        embedding_base_url=None,
        memory_llm_model="test/model",
    )
    parsed = Mem0VectorStoreConfig(**json.loads(json.dumps(config["vector_store"])))

    adapter = Mem0JasperVectorStore(**parsed.config.model_dump())

    assert isinstance(adapter.store, StubStore)
    assert adapter.store.config.qdrant == runtime.qdrant
    assert adapter.config.backend == "qdrant"
