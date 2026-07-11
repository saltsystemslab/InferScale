from __future__ import annotations

import json
from pathlib import Path

import pytest

from locomo_jasper_bench.throughput.config import (
    ALL_CONDITIONS,
    DEFAULT_USER_COUNTS,
    condition_vector_backend,
    parse_args,
    parse_user_counts,
)


def test_parse_user_counts_preserves_order_and_accepts_common_separators() -> None:
    assert parse_user_counts("10, 25;50 100") == (10, 25, 50, 100)


@pytest.mark.parametrize("value", ["", "0", "-5", "10,10", "10:512"])
def test_parse_user_counts_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_user_counts(value)


def test_parse_args_resolves_model_alias_and_user_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_LLAMA", "local/llama-test")
    config, dry_run = parse_args(
        [
            "--model",
            "llama",
            "--conditions",
            "no_memory",
            "mem0_qdrant",
            "--user-counts",
            "2,3",
            "--dataset",
            "data/other.json",
            "--run-id",
            "unit-test",
            "--dry-run",
        ]
    )

    assert config.model == "local/llama-test"
    assert config.model_label == "llama"
    assert config.user_counts == (2, 3)
    assert config.dataset_path == Path("data/other.json")
    assert config.conditions == ("no_memory", "mem0_qdrant")
    assert dry_run is True


def test_default_conditions_are_the_four_way_comparison() -> None:
    config, _ = parse_args(["--model", "test/model", "--dry-run"])

    assert config.conditions == ALL_CONDITIONS
    assert ALL_CONDITIONS == ("no_memory", "mem0_qdrant", "mem0_jasper", "kv_injection")
    assert config.user_counts == DEFAULT_USER_COUNTS
    assert config.context_window == 0
    assert config.vector_distance == "ip"


def test_condition_vector_backends_pair_kv_with_jasper() -> None:
    assert condition_vector_backend("no_memory") is None
    assert condition_vector_backend("mem0_qdrant") == "qdrant"
    assert condition_vector_backend("mem0_jasper") == "jasper"
    assert condition_vector_backend("kv_injection") == "jasper"


def test_context_window_is_configurable_and_validated() -> None:
    config, _ = parse_args(["--model", "test/model", "--context-window", "2"])
    assert config.context_window == 2

    with pytest.raises(SystemExit):
        parse_args(["--model", "test/model", "--context-window", "-1"])


def test_parse_args_rejects_removed_memory_llm_options() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", "test/model", "--memory-llm-model", "other/model"])
    with pytest.raises(SystemExit):
        parse_args(["--model", "test/model", "--memory-llm-base-url", "http://x/v1"])


def test_worker_config_restores_redacted_api_key_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = parse_args(
        [
            "--model",
            "test/model",
            "--conditions",
            "mem0_qdrant",
            "--user-counts",
            "2",
            "--context-window",
            "1",
            "--embedding-api-key",
            "secret-value",
        ]
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_jsonable()), encoding="utf-8")
    monkeypatch.setenv("LOCOMO_THROUGHPUT_EMBEDDING_API_KEY", "secret-value")

    restored = type(config).from_json_file(path)

    assert restored.embedding_api_key == "secret-value"
    assert restored.user_counts == (2,)
    assert restored.context_window == 1
    assert restored.dataset_path == config.dataset_path
    assert "secret-value" not in path.read_text(encoding="utf-8")
