from __future__ import annotations

import json
from pathlib import Path

import pytest

from locomo_jasper_bench.throughput.config import (
    BenchmarkPoint,
    parse_args,
    parse_matrix,
)


def test_parse_matrix_preserves_order_and_accepts_common_separators() -> None:
    assert parse_matrix("10:512, 25x1024;50X2048") == (
        BenchmarkPoint(10, 512),
        BenchmarkPoint(25, 1024),
        BenchmarkPoint(50, 2048),
    )


@pytest.mark.parametrize("value", ["", "10", "0:512", "10:-1", "10:512,10:512"])
def test_parse_matrix_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_matrix(value)


def test_parse_args_resolves_model_alias_and_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_LLAMA", "local/llama-test")
    config, dry_run = parse_args(
        [
            "--model",
            "llama",
            "--conditions",
            "no_memory",
            "prompt_injection",
            "--matrix",
            "2:512,3:1024",
            "--run-id",
            "unit-test",
            "--dry-run",
        ]
    )

    assert config.model == "local/llama-test"
    assert config.model_label == "llama"
    assert config.matrix == (BenchmarkPoint(2, 512), BenchmarkPoint(3, 1024))
    assert config.conditions == ("no_memory", "prompt_injection")
    assert config.vector_backend == "qdrant"
    assert dry_run is True


def test_parse_args_rejects_unaligned_kv_memory() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--conditions",
                "kv_injection",
                "--matrix",
                "2:513",
            ]
        )


def test_worker_config_restores_redacted_api_key_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = parse_args(
        [
            "--model",
            "test/model",
            "--conditions",
            "mem0",
            "--matrix",
            "2:512",
            "--embedding-api-key",
            "secret-value",
        ]
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_jsonable()), encoding="utf-8")
    monkeypatch.setenv("LOCOMO_THROUGHPUT_EMBEDDING_API_KEY", "secret-value")

    restored = type(config).from_json_file(path)

    assert restored.embedding_api_key == "secret-value"
    assert "secret-value" not in path.read_text(encoding="utf-8")
