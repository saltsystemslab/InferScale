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
            "mem0_jasper",
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
    assert config.conditions == ("mem0_jasper", "mem0_qdrant")
    assert dry_run is True


def test_default_conditions_are_the_three_way_comparison() -> None:
    config, _ = parse_args(["--model", "test/model", "--dry-run"])

    assert config.conditions == ALL_CONDITIONS
    assert ALL_CONDITIONS == ("mem0_qdrant", "mem0_jasper", "kv_injection")
    assert config.user_counts == DEFAULT_USER_COUNTS
    assert config.memory_llm_provider == "vllm"
    assert config.memory_llm_model == config.model
    assert "vector_distance" not in config.to_jsonable()


def test_parse_args_rejects_removed_vector_distance_option() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", "test/model", "--vector-distance", "ip"])


def test_parse_args_rejects_removed_memory_llm_model_option() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", "test/model", "--memory-llm-model", "other/model"])


def test_memory_llm_model_always_matches_the_answer_model(tmp_path: Path) -> None:
    config, _ = parse_args(["--model", "test/model"])
    assert config.memory_llm_model == "test/model"

    payload = config.to_jsonable()
    payload["memory_llm_model"] = "other/model"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="always uses the answer model"):
        type(config).from_json_file(path)


def test_condition_vector_backends_pair_kv_with_jasper() -> None:
    assert condition_vector_backend("mem0_qdrant") == "qdrant"
    assert condition_vector_backend("mem0_jasper") == "jasper"
    assert condition_vector_backend("kv_injection") == "jasper"


def test_parse_args_rejects_removed_prompt_injection_condition() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--conditions", "prompt_injection"])


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
    assert restored.dataset_path == config.dataset_path
    assert restored.memory_llm_cache_dir == config.memory_llm_cache_dir
    assert "secret-value" not in path.read_text(encoding="utf-8")


def test_worker_config_rejects_removed_vector_distance_field(tmp_path: Path) -> None:
    config, _ = parse_args(["--model", "test/model"])
    payload = config.to_jsonable()
    payload["vector_distance"] = "ip"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="vector_distance"):
        type(config).from_json_file(path)


def test_parse_args_kv_store_backend_round_trips(tmp_path: Path) -> None:
    config, _ = parse_args(
        [
            "--model",
            "test/model",
            "--kv-store-backend",
            "cpu-pinned",
            "--kv-staging-slots",
            "8",
        ]
    )

    assert config.kv_store_backend == "cpu-pinned"
    assert config.kv_staging_slots == 8

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_jsonable()), encoding="utf-8")
    restored = type(config).from_json_file(path)
    assert restored.kv_store_backend == "cpu-pinned"
    assert restored.kv_staging_slots == 8


def test_parse_args_defaults_to_gpu_store_with_prefix_caching() -> None:
    config, _ = parse_args(["--model", "test/model"])

    assert config.kv_store_backend == "gpu"
    assert config.kv_staging_slots == 4
    assert config.kv_enable_prefix_caching is True


def test_parse_args_no_kv_prefix_caching_flag() -> None:
    config, _ = parse_args(["--model", "test/model", "--no-kv-prefix-caching"])

    assert config.kv_enable_prefix_caching is False


def test_parse_args_rejects_non_positive_staging_slots() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", "test/model", "--kv-staging-slots", "0"])


def test_parse_args_defaults_to_topk50_window50() -> None:
    config, _ = parse_args(["--model", "test/model"])

    assert config.top_k == 50
    assert config.context_window == 50


def test_parse_args_context_window_round_trips(tmp_path: Path) -> None:
    config, _ = parse_args(["--model", "test/model", "--context-window", "0"])

    assert config.context_window == 0

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_jsonable()), encoding="utf-8")
    restored = type(config).from_json_file(path)
    assert restored.context_window == 0


def test_parse_args_rejects_negative_context_window() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", "test/model", "--context-window", "-1"])


def test_cli_overrides_env_prefix_caching_in_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCOMO_KV_ENABLE_PREFIX_CACHING", "0")
    config, _ = parse_args(["--model", "test/model", "--kv-prefix-caching"])
    assert config.kv_enable_prefix_caching is True

    monkeypatch.setenv("LOCOMO_KV_ENABLE_PREFIX_CACHING", "1")
    config, _ = parse_args(["--model", "test/model", "--no-kv-prefix-caching"])
    assert config.kv_enable_prefix_caching is False
