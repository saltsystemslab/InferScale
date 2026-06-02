from pathlib import Path

from locomo_jasper_bench.config import parse_args


def test_context_mode_defaults_to_mem0():
    config = parse_args([])

    assert config.context_mode == "mem0"


def test_context_mode_accepts_full():
    config = parse_args(["--context-mode", "full"])

    assert config.context_mode == "full"


def test_context_mode_accepts_retrieval_alias():
    config = parse_args(["--context-mode", "retrieval"])

    assert config.context_mode == "mem0"


def test_results_dir_default_uses_benchmark_results_root(monkeypatch):
    monkeypatch.setenv("BENCHMARK_RESULTS_ROOT", "/scratch/tester/benchmark-jasper/results")

    config = parse_args([])

    assert config.results_dir == Path("/scratch/tester/benchmark-jasper/results")
