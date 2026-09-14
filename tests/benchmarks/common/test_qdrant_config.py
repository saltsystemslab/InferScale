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


def test_default_qdrant_connection_uses_runpod_environment_and_exact_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example-pod-6333.proxy.runpod.net")
    settings = RuntimeConfig.from_dict({}, root=tmp_path).qdrant

    assert settings.url == "https://example-pod-6333.proxy.runpod.net"
    assert settings.grpc_port == 6334
    assert settings.prefer_grpc is False
    assert settings.exact is True


@pytest.mark.parametrize("port", ["", "not-a-port", "31234"])
def test_stale_grpc_environment_does_not_affect_rest_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port: str,
) -> None:
    monkeypatch.setenv("QDRANT_GRPC_PORT", port)
    monkeypatch.setenv("QDRANT_URL", "https://example-pod-6333.proxy.runpod.net")
    config = RuntimeConfig.from_dict({}, root=tmp_path).qdrant
    connection = config.client_kwargs()
    assert connection["prefer_grpc"] is False
    assert connection["grpc_port"] == 6334
    assert connection["url"] == "https://example-pod-6333.proxy.runpod.net"


@pytest.mark.parametrize("url", [
    "https://example-pod-6333.proxy.runpod.net",
    "https://EXAMPLE-POD-6333.PROXY.RUNPOD.NET./",
])
def test_grpc_rejects_runpod_https_proxy_at_connection_use(url: str) -> None:
    config = QdrantConfig(url=url, prefer_grpc=True)
    with pytest.raises(ValueError, match=r"direct host URL.*qdrant\.grpc_port"):
        config.client_kwargs()
    assert QdrantConfig(url=url, prefer_grpc=False).client_kwargs()["url"] == url


def test_missing_endpoint_only_fails_when_qdrant_connection_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    runtime = RuntimeConfig.from_dict({}, root=tmp_path)
    data = {"model": "test/model", "inferscale": {}}
    assert MemoryRunConfig.from_dict(data, runtime=runtime).qdrant.url == ""
    assert ThroughputConfig.from_dict(data, runtime=runtime).qdrant.url == ""
    with pytest.raises(ValueError, match="Set QDRANT_URL to your Runpod CPU Pod"):
        runtime.qdrant.client_kwargs()


def test_runpod_https_request_preserves_proxy_port_and_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    from qdrant_client import QdrantClient

    monkeypatch.setenv("QDRANT_URL", "https://example-pod-6333.proxy.runpod.net")
    monkeypatch.setenv("QDRANT_API_KEY", "test-secret")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"result": {"collections": []}, "status": "ok", "time": 0})

    client = QdrantClient(
        **QdrantConfig().client_kwargs(),
        check_compatibility=False,
        transport=httpx.MockTransport(respond),
    )
    try:
        assert client.get_collections().collections == []
    finally:
        client.close()
    assert len(requests) == 1
    assert str(requests[0].url) == "https://example-pod-6333.proxy.runpod.net/collections"
    assert requests[0].headers["api-key"] == "test-secret"


def test_grpc_collections_uses_tls_mapped_port_and_auth_without_http_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qdrant_client import QdrantClient, grpc

    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "test-secret")
    channels: list[dict[str, Any]] = []
    calls: list[tuple[str, Any, int]] = []

    class Channel:
        closed = False

        def unary_unary(self, method: str, **kwargs: Any) -> Any:
            def call(request: Any, timeout: int) -> Any:
                calls.append((method, request, timeout))
                assert method == "/qdrant.Collections/List"
                return grpc.ListCollectionsResponse(
                    collections=[grpc.CollectionDescription(name="existing-collection")],
                )
            return call

        def close(self) -> None:
            self.closed = True

    channel = Channel()

    def get_channel(**kwargs: Any) -> Channel:
        channels.append(kwargs)
        return channel

    def unexpected_http(*args: Any, **kwargs: Any) -> None:
        pytest.fail("gRPC-only connections must not probe HTTP")

    monkeypatch.setattr("qdrant_client.qdrant_remote.get_channel", get_channel)
    monkeypatch.setattr("qdrant_client.qdrant_remote.Thread", unexpected_http)
    monkeypatch.setattr("httpx.Client.send", unexpected_http)
    client = QdrantClient(
        **QdrantConfig(prefer_grpc=True, grpc_port=31234, timeout=17).client_kwargs(),
    )
    try:
        assert [item.name for item in client.get_collections().collections] == ["existing-collection"]
    finally:
        client.close()

    assert channels
    for options in channels:
        assert options["host"] == "qdrant.example"
        assert options["port"] == 31234
        assert options["ssl"] is True
        assert ("api-key", "test-secret") in options["metadata"]
    assert len(calls) == 1
    assert isinstance(calls[0][1], grpc.ListCollectionsRequest)
    assert calls[0][2] == 17
    assert channel.closed


def test_api_key_is_resolved_at_connection_time_and_never_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "original-secret")
    runtime = RuntimeConfig.from_dict({}, root=tmp_path)
    data = {"model": "test/model", "inferscale": {}}
    memory = MemoryRunConfig.from_dict(data, runtime=runtime)
    throughput = ThroughputConfig.from_dict(data, runtime=runtime)

    monkeypatch.setenv("QDRANT_API_KEY", "rotated-secret")
    assert runtime.qdrant.client_kwargs()["api_key"] == "rotated-secret"
    for config in (runtime.qdrant.to_dict(), memory.to_jsonable(), throughput.to_jsonable()):
        serialized = json.dumps(config)
        assert "original-secret" not in serialized
        assert "rotated-secret" not in serialized
        assert "QDRANT_API_KEY" not in serialized
    assert "secret" not in repr(runtime.qdrant)
    monkeypatch.delenv("QDRANT_API_KEY")
    assert runtime.qdrant.client_kwargs()["api_key"] is None


@pytest.mark.parametrize("settings", [
    {"url": ":memory:"},
    {"url": "file:///tmp/qdrant"},
    {"url": "http://user:secret@qdrant.example"},
    {"url": "http://qdrant.example?api_key=secret"},
    {"url": "http://qdrant.example:65536"},
    {"url": "http://qdrant.example#fragment"},
    {"url": None},
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
    {"api_key": "secret-must-stay-in-environment"},
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
    tmp_path: Path, runtime: RuntimeConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://different-qdrant.example")
    config = ThroughputConfig.from_dict({"model": "test/model", "inferscale": {}}, runtime=runtime)
    snapshot = tmp_path / "worker.json"
    snapshot.write_text(json.dumps(config.to_jsonable()), encoding="utf-8")

    restored = ThroughputConfig.from_json_file(snapshot)

    assert restored.qdrant == runtime.qdrant
    assert isinstance(restored.qdrant, QdrantConfig)
    assert _vector_config(restored, "qdrant").qdrant == runtime.qdrant


def test_worker_snapshot_preserves_environment_endpoint_after_environment_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://original-pod-6333.proxy.runpod.net")
    runtime = RuntimeConfig.from_dict({}, root=tmp_path)
    config = ThroughputConfig.from_dict({"model": "test/model", "inferscale": {}}, runtime=runtime)
    snapshot = tmp_path / "worker.json"
    snapshot.write_text(json.dumps(config.to_jsonable()), encoding="utf-8")
    monkeypatch.setenv("QDRANT_URL", "https://different-pod-6333.proxy.runpod.net")

    restored = ThroughputConfig.from_json_file(snapshot)

    assert restored.qdrant.url == "https://original-pod-6333.proxy.runpod.net"
    assert restored.qdrant == runtime.qdrant
    assert QdrantConfig().url == "https://different-pod-6333.proxy.runpod.net"


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
