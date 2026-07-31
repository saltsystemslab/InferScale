from __future__ import annotations

from pathlib import Path

import pytest

from rag_bench.config import RagBenchConfig, parse_args


def test_defaults() -> None:
    config = parse_args(["--skip-judge", "--model", "llama"])

    assert config.dataset_name == "multihoprag"
    assert config.data_dir == Path("data/multihoprag")
    assert config.chunk_size == 1024
    assert config.context_window == 5
    assert config.top_k == 15
    assert config.answer_backend == "vllm-kv"
    assert config.result_mode() == "rag-kv"
    assert config.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.kv_dtype == "bfloat16"
    assert config.kv_block_size == 16
    assert config.kv_store_backend == "cpu"
    assert config.skip_judge is True and config.judge_provider == "none"


def test_prefix_backend_mode_string() -> None:
    config = parse_args(["--skip-judge", "--answer-backend", "vllm-prefix"])

    assert config.result_mode() == "rag-prefix"


def test_model_alias_resolution_via_shared_registry() -> None:
    config = parse_args(["--skip-judge", "--answer-model", "qwen"])

    assert config.model == "Qwen/Qwen2.5-7B-Instruct"


def test_data_dir_follows_dataset_name() -> None:
    config = RagBenchConfig(dataset_name="qasper")

    assert config.data_dir == Path("data/qasper")


def test_to_jsonable_redacts_secrets_and_stringifies_paths(tmp_path) -> None:
    config = parse_args(
        [
            "--skip-judge",
            "--results-dir",
            str(tmp_path),
            "--embedding-api-key",
            "sk-secret",
            "--kv-chunk-cache-root",
            str(tmp_path / "kv"),
        ]
    )

    data = config.to_jsonable()

    assert data["embedding_api_key"] == "<redacted>"
    assert data["results_dir"] == str(tmp_path)
    assert data["kv_chunk_cache_root"] == str(tmp_path / "kv")
    assert isinstance(data["data_dir"], str)
    assert data["mode"] == "rag-kv"
    assert data["kv_store_backend"] == "cpu"
    assert data["jasper_effective_beam_width"] == 64


def test_run_dir_property(tmp_path) -> None:
    config = parse_args(["--skip-judge", "--results-dir", str(tmp_path), "--run-id", "abc"])

    assert config.run_dir == tmp_path / "abc"


def test_top_k_and_chunk_size_validation() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--top-k", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--chunk-size", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--context-window", "-1"])


def test_jasper_beam_cap_validation() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--jasper-beam-width", "960"])


def test_memory_budget_validation_rejects_oversized_grids() -> None:
    # 50 x 1024 + margin exceeds min(32768, 32768 - answer tokens).
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--top-k", "50"])
    # Raising the limits makes the same grid parse.
    config = parse_args(
        [
            "--skip-judge",
            "--top-k",
            "50",
            "--kv-max-position",
            "65536",
            "--kv-max-model-len",
            "65536",
        ]
    )
    assert config.top_k == 50


def test_prefix_requires_prefix_caching() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--skip-judge",
                "--answer-backend",
                "vllm-prefix",
                "--no-kv-prefix-caching",
            ]
        )


def test_judge_flag_interactions() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--judge", "vllm"])
    with pytest.raises(SystemExit):
        parse_args(["--rejudge"])
    with pytest.raises(SystemExit):
        parse_args(["--judge-only", "--judge", "none"])
    config = parse_args(["--judge-only", "--judge", "vllm", "--judge-model", "j"])
    assert config.judge_only and config.judge_provider == "vllm"
    assert config.judge_model == "j"


def test_max_queries_and_stage_flags() -> None:
    config = parse_args(["--skip-judge", "--max-queries", "25", "--estimate-only"])

    assert config.max_queries == 25
    assert config.estimate_only is True
    assert config.preembed_only is False
