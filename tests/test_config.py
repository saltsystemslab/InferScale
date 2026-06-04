from pathlib import Path

import pytest

from locomo_jasper_bench.config import BenchmarkConfig, parse_args


def test_context_mode_defaults_to_mem0():
    config = parse_args([])

    assert config.context_mode == "mem0"


def test_context_mode_accepts_full():
    config = parse_args(["--context-mode", "full"])

    assert config.context_mode == "full"


def test_context_mode_accepts_retrieval_alias():
    config = parse_args(["--context-mode", "retrieval"])

    assert config.context_mode == "mem0"


def test_vector_backend_accepts_qdrant():
    config = parse_args(["--vector-backend", "qdrant"])

    assert config.vector_backend == "qdrant"


def test_jasper_alpha_defaults_to_one():
    config = parse_args([])

    assert config.jasper_alpha == 1.0


def test_jasper_alpha_can_be_overridden():
    config = parse_args(["--jasper-alpha", "0.75"])

    assert config.jasper_alpha == 0.75


def test_results_dir_default_uses_benchmark_results_root(monkeypatch):
    monkeypatch.setenv("BENCHMARK_RESULTS_ROOT", "/scratch/tester/benchmark-jasper/results")

    config = parse_args([])

    assert config.results_dir == Path("/scratch/tester/benchmark-jasper/results")


def test_embedding_cache_dir_default_uses_benchmark_cache_root(monkeypatch):
    monkeypatch.setenv("BENCHMARK_CACHE_ROOT", "/scratch/tester/benchmark-jasper/cache")

    config = parse_args([])

    assert config.embedding_cache_enabled is True
    assert config.embedding_cache_dir == Path("/scratch/tester/benchmark-jasper/cache/embeddings")


def test_embedding_cache_can_be_disabled():
    config = parse_args(["--no-embedding-cache"])

    assert config.embedding_cache_enabled is False


def test_config_to_jsonable_redacts_api_keys():
    config = BenchmarkConfig(
        llm_api_key="llm-secret",
        judge_api_key="judge-secret",
        embedding_api_key="embed-secret",
    )

    data = config.to_jsonable()

    assert data["llm_api_key"] == "<redacted>"
    assert data["judge_api_key"] == "<redacted>"
    assert data["embedding_api_key"] == "<redacted>"
