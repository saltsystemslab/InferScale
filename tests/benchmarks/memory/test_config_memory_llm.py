from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common.config import ConfigError
from memory_config import make_memory_config, memory_runtime


@pytest.fixture(autouse=True)
def clear_connection_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EXTRACTION_LLM_BASE_URL", "EXTRACTION_LLM_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "JUDGE_LLM_BASE_URL", "JUDGE_LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("provider", ["vllm", "none"])
def test_supported_judge_providers_remain_available(provider: str) -> None:
    config = make_memory_config(judge={'provider': provider})

    assert config.judge_provider == provider
    assert config.skip_judge is (provider == "none")


def test_openai_judge_provider_is_rejected() -> None:
    with pytest.raises(ConfigError, match="provider must be vllm or none"):
        make_memory_config(judge={'provider': 'openai'})


def test_old_judge_provider_environment_does_not_override_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")

    assert make_memory_config().judge_provider == "vllm"


def test_answer_and_judge_token_budgets_remain_configurable() -> None:
    assert make_memory_config().max_answer_tokens == 512
    assert make_memory_config().max_judge_tokens == 4
    assert make_memory_config(inferscale={'generation': {'max_tokens': 256}}).max_answer_tokens == 256
    assert make_memory_config(judge={'max_tokens': 256}).max_judge_tokens == 256


def test_preembed_workers_come_from_json_and_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCOMO_PREEMBED_WORKERS", "8")
    assert make_memory_config().preembed_workers == 4
    assert make_memory_config(preembed_workers=8).preembed_workers == 8
    assert make_memory_config(preembed_workers=2).preembed_workers == 2

    with pytest.raises(ConfigError, match="preembed_workers must be >= 1"):
        make_memory_config(preembed_workers=0)


def test_memory_llm_defaults_and_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    config = make_memory_config(runtime=memory_runtime(storage={"cache_root": str(cache_root)}))

    assert config.memory_llm_provider == "vllm"
    assert config.memory_llm_model == config.model
    assert config.memory_llm_base_url == "http://localhost:8000/v1"
    assert config.memory_llm_api_key is None
    assert config.memory_llm_cache_dir == cache_root / "mem0-inference"


def test_memory_llm_model_is_the_resolved_answer_model() -> None:
    alias_config = make_memory_config(model="llama")
    assert alias_config.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert alias_config.memory_llm_model == alias_config.model

    raw_config = make_memory_config(model="example-org/custom-model")
    assert raw_config.memory_llm_model == "example-org/custom-model"


def test_memory_config_rejects_separate_extraction_model() -> None:
    with pytest.raises(ConfigError, match="memory_llm_model"):
        make_memory_config(model="answer/model", memory_llm_model="other/model")
    with pytest.raises(ConfigError, match="llm_model"):
        make_memory_config(mem0={'llm_model': 'other/model'})


def test_memory_llm_ignores_openai_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-fallback.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-secret")
    config = make_memory_config()

    assert config.memory_llm_base_url == "http://localhost:8000/v1"
    assert config.memory_llm_api_key is None


def test_memory_llm_credentials_use_environment_but_endpoint_uses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-fallback.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-secret")
    monkeypatch.setenv("EXTRACTION_LLM_BASE_URL", "https://memory-env.example/v1")
    monkeypatch.setenv("EXTRACTION_LLM_API_KEY", "memory-env-secret")
    config = make_memory_config()

    assert config.memory_llm_base_url == "http://localhost:8000/v1"
    assert config.memory_llm_api_key == "memory-env-secret"


def test_memory_llm_json_precedes_environment_and_redacts_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXTRACTION_LLM_BASE_URL", "https://memory-env.example/v1")
    monkeypatch.setenv("EXTRACTION_LLM_API_KEY", "memory-env-secret")
    cache_root = tmp_path / "cache"
    config = make_memory_config(runtime=memory_runtime(storage={'cache_root': str(cache_root)}), mem0={'llm_base_url': 'https://memory-override.example/v1'})

    assert config.memory_llm_base_url == "https://memory-override.example/v1"
    assert config.memory_llm_api_key == "memory-env-secret"
    assert config.memory_llm_cache_dir == cache_root / "mem0-inference"

    serialized = config.to_jsonable()
    assert serialized["memory_llm_provider"] == "vllm"
    assert serialized["memory_llm_model"] == config.model
    assert serialized["memory_llm_api_key"] == "<redacted>"
    assert serialized["memory_llm_cache_dir"] == str(cache_root / "mem0-inference")
