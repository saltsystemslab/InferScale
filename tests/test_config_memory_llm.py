from __future__ import annotations

from pathlib import Path

import pytest

from locomo_jasper_bench.config import BenchmarkConfig, parse_args


MEMORY_LLM_ENV_VARS = (
    "MEM0_LLM_BASE_URL",
    "MEM0_LLM_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "LOCOMO_VLLM_MODEL",
    "LOCOMO_MODEL_LLAMA",
    "MODEL_LLAMA",
)


def _clear_memory_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MEMORY_LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("provider", ["vllm", "none"])
def test_supported_judge_providers_remain_available(provider: str) -> None:
    config = parse_args(["--judge", provider])

    assert config.judge_provider == provider
    assert config.skip_judge is (provider == "none")


def test_openai_judge_cli_provider_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        parse_args(["--judge", "openai"])

    assert "invalid choice: 'openai'" in capsys.readouterr().err


def test_openai_judge_environment_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")

    with pytest.raises(SystemExit, match="2"):
        parse_args([])

    assert "JUDGE_PROVIDER must be vllm or none" in capsys.readouterr().err


def test_structured_judge_token_budget_defaults_to_512_and_remains_configurable() -> None:
    assert parse_args(["--judge", "none"]).max_answer_tokens == 512
    assert parse_args(["--judge", "none"]).max_judge_tokens == 4
    assert parse_args(["--judge", "none", "--max-answer-tokens", "256"]).max_answer_tokens == 256
    assert parse_args(["--judge", "none", "--max-judge-tokens", "256"]).max_judge_tokens == 256


def test_memory_llm_defaults_and_cache_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_memory_llm_environment(monkeypatch)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("BENCHMARK_CACHE_ROOT", str(cache_root))

    config = parse_args(["--judge", "none"])

    assert config.memory_llm_provider == "vllm"
    assert config.memory_llm_model == config.model
    assert config.memory_llm_base_url is None
    assert config.memory_llm_api_key is None
    assert config.memory_llm_cache_dir == cache_root / "mem0-inference"


def test_memory_llm_model_is_the_resolved_answer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_memory_llm_environment(monkeypatch)

    alias_config = parse_args(["--judge", "none", "--answer-model", "llama"])
    assert alias_config.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert alias_config.memory_llm_model == alias_config.model

    raw_config = parse_args(["--judge", "none", "--answer-model", "example-org/custom-model"])
    assert raw_config.memory_llm_model == "example-org/custom-model"


def test_memory_llm_model_flag_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        parse_args(["--judge", "none", "--memory-llm-model", "some/model"])

    assert "unrecognized arguments: --memory-llm-model" in capsys.readouterr().err


def test_benchmark_config_rejects_mismatched_memory_llm_model() -> None:
    with pytest.raises(ValueError, match="always uses the answer model"):
        BenchmarkConfig(model="answer/model", memory_llm_model="other/model")


def test_memory_llm_ignores_openai_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_memory_llm_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-fallback.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-secret")

    config = parse_args(["--judge", "none"])

    assert config.memory_llm_base_url is None
    assert config.memory_llm_api_key is None


def test_memory_llm_specific_environment_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_memory_llm_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-fallback.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-secret")
    monkeypatch.setenv("MEM0_LLM_BASE_URL", "https://memory-env.example/v1")
    monkeypatch.setenv("MEM0_LLM_API_KEY", "memory-env-secret")

    config = parse_args(["--judge", "none"])

    assert config.memory_llm_base_url == "https://memory-env.example/v1"
    assert config.memory_llm_api_key == "memory-env-secret"


def test_memory_llm_cli_overrides_environment_and_redacts_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_memory_llm_environment(monkeypatch)
    monkeypatch.setenv("MEM0_LLM_BASE_URL", "https://memory-env.example/v1")
    monkeypatch.setenv("MEM0_LLM_API_KEY", "memory-env-secret")
    cache_dir = tmp_path / "inference-cache"

    config = parse_args(
        [
            "--judge",
            "none",
            "--memory-llm-base-url",
            "https://memory-cli.example/v1",
            "--memory-llm-api-key",
            "memory-cli-secret",
            "--memory-llm-cache-dir",
            str(cache_dir),
        ]
    )

    assert config.memory_llm_base_url == "https://memory-cli.example/v1"
    assert config.memory_llm_api_key == "memory-cli-secret"
    assert config.memory_llm_cache_dir == cache_dir

    serialized = config.to_jsonable()
    assert serialized["memory_llm_provider"] == "vllm"
    assert serialized["memory_llm_model"] == config.model
    assert serialized["memory_llm_api_key"] == "<redacted>"
    assert serialized["memory_llm_cache_dir"] == str(cache_dir)
