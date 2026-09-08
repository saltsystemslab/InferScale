from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.common.cache_identity import endpoint_cache_key
from benchmarks.common.config import (
    ConfigError,
    RuntimeConfig,
    judge_config,
)
from benchmarks.memory.config import MemoryRunConfig


ROLE_ENVIRONMENT = (
    "JUDGE_LLM_BASE_URL",
    "JUDGE_LLM_API_KEY",
    "EXTRACTION_LLM_BASE_URL",
    "EXTRACTION_LLM_API_KEY",
)
LEGACY_ENVIRONMENT = (
    "JUDGE_BASE_URL",
    "JUDGE_API_KEY",
    "MEM0_LLM_BASE_URL",
    "MEM0_LLM_API_KEY",
    "LOCOMO_VLLM_API_KEY",
)


@pytest.fixture(autouse=True)
def clear_connection_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*ROLE_ENVIRONMENT, *LEGACY_ENVIRONMENT, "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def runtime(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig.from_dict(
        {
            "models": {"llama": "meta-llama/Llama-3.2-3B-Instruct"},
            "judge_server": {
                "base_url": "http://runtime-judge.example:8000/v1",
                "api_key": "runtime-judge-token",
            },
        },
        root=tmp_path,
    )


@pytest.fixture
def memory_data() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "configs/memory/accuracy-latency/llama.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", LEGACY_ENVIRONMENT)
def test_legacy_names_do_not_configure_connections(
    monkeypatch: pytest.MonkeyPatch,
    runtime: RuntimeConfig,
    memory_data: dict[str, Any],
    name: str,
) -> None:
    value = "http://legacy.example:9999/v1" if name.endswith("URL") else "legacy-token"
    monkeypatch.setenv(name, value)
    config = MemoryRunConfig.from_dict(memory_data, runtime=runtime)

    assert config.judge.base_url == runtime.judge_server.base_url
    assert config.judge.api_key == "runtime-judge-token"
    assert config.memory_llm_base_url == "http://localhost:8000/v1"
    assert config.memory_llm_api_key is None
    assert config.embedding_api_key is None


@pytest.mark.parametrize("blank_environment", (False, True))
def test_default_connections_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    memory_data: dict[str, Any],
    blank_environment: bool,
) -> None:
    if blank_environment:
        for name in ROLE_ENVIRONMENT:
            monkeypatch.setenv(name, "")
    memory_data.pop("mem0")
    memory_data.pop("judge")
    config = MemoryRunConfig.from_dict(memory_data, runtime=RuntimeConfig.from_dict({}, root=tmp_path))

    assert config.judge.base_url == "http://127.0.0.1:8000/v1"
    assert config.judge.api_key == "token-abc123"
    assert config.memory_llm_base_url == "http://localhost:8000/v1"
    assert config.memory_llm_api_key is None


@pytest.mark.parametrize("section_url", [None, "https://json.example/v1"])
def test_shared_judge_endpoint_comes_from_json(
    monkeypatch: pytest.MonkeyPatch, runtime: RuntimeConfig, section_url: str | None,
) -> None:
    monkeypatch.setenv("JUDGE_LLM_BASE_URL", "https://env.example/v1")
    section = {"base_url": section_url} if section_url is not None else {}
    config = judge_config(section, runtime, where="test judge")
    assert config.base_url == (section_url or runtime.judge_server.base_url)


def test_json_connections_ignore_environment_settings_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch, runtime: RuntimeConfig, memory_data: dict[str, Any],
) -> None:
    memory_data["judge"]["base_url"] = "https://json-judge.example/v1"
    memory_data["mem0"]["llm_base_url"] = "https://json-extraction.example/v1"
    original_data = json.dumps(memory_data, sort_keys=True)
    monkeypatch.setenv("JUDGE_LLM_BASE_URL", "https://env-judge.example/v1")
    monkeypatch.setenv("EXTRACTION_LLM_BASE_URL", "https://env-extraction.example/v1")
    config = MemoryRunConfig.from_dict(memory_data, runtime=runtime)
    assert config.judge.base_url == "https://json-judge.example/v1"
    assert config.memory_llm_base_url == "https://json-extraction.example/v1"
    assert json.dumps(memory_data, sort_keys=True) == original_data


def test_json_connections_are_used_without_environment_overrides(
    runtime: RuntimeConfig, memory_data: dict[str, Any]
) -> None:
    memory_data["judge"]["base_url"] = "https://json-judge.example/v1"
    memory_data["mem0"]["llm_base_url"] = "https://json-extraction.example/v1"
    config = MemoryRunConfig.from_dict(memory_data, runtime=runtime)

    assert config.judge.base_url == "https://json-judge.example/v1"
    assert config.memory_llm_base_url == "https://json-extraction.example/v1"


@pytest.mark.parametrize("role", ("embedding", "judge", "extraction"))
def test_credentials_and_endpoints_are_isolated_by_role(
    monkeypatch: pytest.MonkeyPatch,
    runtime: RuntimeConfig,
    memory_data: dict[str, Any],
    role: str,
) -> None:
    prefix = {"embedding": "OPENAI", "judge": "JUDGE_LLM", "extraction": "EXTRACTION_LLM"}[role]
    monkeypatch.setenv(f"{prefix}_API_KEY", f"{role}-token")
    monkeypatch.setenv(f"{prefix}_BASE_URL", f"https://{role}.example/v1")
    config = MemoryRunConfig.from_dict(memory_data, runtime=runtime)

    assert config.embedding_api_key == ("embedding-token" if role == "embedding" else None)
    assert config.judge.api_key == ("judge-token" if role == "judge" else "runtime-judge-token")
    assert config.memory_llm_api_key == ("extraction-token" if role == "extraction" else None)
    assert config.judge.base_url == runtime.judge_server.base_url
    assert config.memory_llm_base_url == "http://localhost:8000/v1"
    assert config.embedding_base_url is None


def test_serialized_connections_redact_all_credentials(
    monkeypatch: pytest.MonkeyPatch, runtime: RuntimeConfig, memory_data: dict[str, Any]
) -> None:
    secrets = {
        "OPENAI_API_KEY": "private-embedding-token",
        "JUDGE_LLM_API_KEY": "private-judge-token",
        "EXTRACTION_LLM_API_KEY": "private-extraction-token",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    config = MemoryRunConfig.from_dict(memory_data, runtime=runtime)
    serialized = config.to_jsonable()

    assert serialized["embedding_api_key"] == "<redacted>"
    assert serialized["judge"]["api_key"] == "<redacted>"
    assert serialized["memory_llm_api_key"] == "<redacted>"
    assert all(secret not in json.dumps(serialized) for secret in secrets.values())
    assert runtime.judge_api_key() == "private-judge-token"


def test_extraction_port_validation_uses_json_endpoint(
    runtime: RuntimeConfig, memory_data: dict[str, Any],
) -> None:
    memory_data["mem0"]["llm_base_url"] = "http://extraction.example:9001/v1"
    with pytest.raises(ConfigError, match=r"port 9001 must equal extraction.port 8000"):
        MemoryRunConfig.from_dict(memory_data, runtime=runtime)
    memory_data["extraction"]["port"] = 9001
    config = MemoryRunConfig.from_dict(memory_data, runtime=runtime)
    assert config.memory_llm_base_url == "http://extraction.example:9001/v1"
    assert config.extraction.port == 9001


def test_extraction_cache_identity_uses_json_endpoint_and_ignores_key_rotation(
    monkeypatch: pytest.MonkeyPatch, runtime: RuntimeConfig, memory_data: dict[str, Any],
) -> None:
    baseline = MemoryRunConfig.from_dict(memory_data, runtime=runtime)
    memory_data["mem0"]["llm_base_url"] = "http://localhost:8000/v1/"
    monkeypatch.setenv("EXTRACTION_LLM_API_KEY", "rotated-token")
    rotated = MemoryRunConfig.from_dict(memory_data, runtime=runtime)
    assert endpoint_cache_key(rotated.memory_llm_base_url) == endpoint_cache_key(baseline.memory_llm_base_url)
    assert rotated.memory_llm_cache_dir == baseline.memory_llm_cache_dir
    memory_data["mem0"]["llm_base_url"] = "http://other-extraction.example:8000/v1"
    other_endpoint = MemoryRunConfig.from_dict(memory_data, runtime=runtime)
    assert endpoint_cache_key(other_endpoint.memory_llm_base_url) != endpoint_cache_key(baseline.memory_llm_base_url)


@pytest.mark.parametrize(("section", "name", "value"), [
    ("engine", "block_size", True), ("engine", "max_model_len", "32768"),
    ("engine", "enable_prefix_caching", "false"), ("kv", "device", 0),
    ("index", "alpha", False), ("embedding", "batch_size", 1.5),
    ("generation", "temperature", "0.1"),
])
def test_nested_json_types_are_validated_before_construction(
    runtime: RuntimeConfig, memory_data: dict[str, Any], section: str, name: str, value: Any,
) -> None:
    memory_data["inferscale"][section][name] = value
    with pytest.raises(ConfigError, match=name):
        MemoryRunConfig.from_dict(memory_data, runtime=runtime)


def test_default_embedding_endpoint_cannot_be_replaced_by_sdk_environment(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace
    from inferscale.v1.embedding.openai import DEFAULT_OPENAI_BASE_URL, OpenAIEmbedder
    from benchmarks.memory.mem0.provider import build_mem0_config
    from benchmarks.common.vector_types import VectorStoreConfig

    captured = {}
    def client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example/v1")
    monkeypatch.setenv("VLLM_BASE_URL", "https://environment-extraction.example/v1")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    OpenAIEmbedder(model="embedding", api_key="test-key")
    assert captured["base_url"] == DEFAULT_OPENAI_BASE_URL
    config = build_mem0_config(
        store_root="unused", vector_config=VectorStoreConfig(backend="jasper"),
        embedding_model="embedding", embedding_api_key="test-key", embedding_base_url=None,
        memory_llm_model="answer-model", memory_llm_base_url=None,
    )
    assert config["embedder"]["config"]["openai_base_url"] == DEFAULT_OPENAI_BASE_URL
    assert config["llm"]["config"]["vllm_base_url"] == "http://localhost:8000/v1"
