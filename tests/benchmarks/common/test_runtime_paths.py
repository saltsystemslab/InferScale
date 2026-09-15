from __future__ import annotations

from pathlib import Path
import os

import pytest

from benchmarks.common.paths import StorageConfig, resolve_layout
from benchmarks.common.config import RuntimeConfig


def test_local_store_scratch_dir_defaults_to_container_local_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    # TMPDIR points at the network volume on remote hosts, which cannot host sqlite.
    monkeypatch.setenv("TMPDIR", "/workspace/tmp")
    layout = resolve_layout(StorageConfig(runtime_root=Path("/workspace")))

    assert layout.local_store_scratch_dir("run-1") == Path("/tmp/inferscale-bench-stores/run-1")


def test_local_store_scratch_dir_honors_storage_override() -> None:
    layout = resolve_layout(StorageConfig(local_store_dir=Path("/mnt/nvme")))

    assert layout.local_store_scratch_dir("run-1") == Path("/mnt/nvme/inferscale-bench-stores/run-1")


def test_local_store_scratch_dir_separates_runs() -> None:
    layout = resolve_layout(StorageConfig())

    assert layout.local_store_scratch_dir("run-a") != layout.local_store_scratch_dir("run-b")


def test_project_local_and_explicit_runtime_roots(tmp_path: Path) -> None:
    local = resolve_layout(StorageConfig(), root=tmp_path)
    remote = resolve_layout(StorageConfig(runtime_root=Path("/workspace")), root=tmp_path)

    assert local.cache_root == tmp_path / ".cache"
    assert local.results_root == tmp_path / "results"
    assert local.memory_llm_cache_dir == tmp_path / ".cache" / "mem0-inference"
    assert remote.cache_root == Path("/workspace/.cache")
    assert remote.tmp_dir == Path("/workspace/tmp")


def test_explicit_storage_paths_override_runtime_root(tmp_path: Path) -> None:
    layout = resolve_layout(
        StorageConfig(runtime_root=Path("/workspace"), cache_root=tmp_path / "cache", tmp_dir=tmp_path / "tmp")
    )

    assert layout.embedding_cache_dir == tmp_path / "cache" / "embeddings"
    assert layout.mem0_dir == tmp_path / "cache" / "mem0"
    assert layout.environment()["TMPDIR"] == str(tmp_path / "tmp")


def test_runtime_json_controls_paths_and_exported_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("BENCHMARK_RUNTIME_ROOT", "/ignored/runtime")
    monkeypatch.setenv("BENCHMARK_RESULTS_ROOT", "/ignored/results")
    monkeypatch.setenv("HF_HOME", "/ignored/huggingface")
    monkeypatch.setenv("MEM0_DIR", "/ignored/mem0")
    monkeypatch.setenv("VLLM_NO_USAGE_STATS", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    runtime = RuntimeConfig.from_dict({
        "storage": {"runtime_root": str(tmp_path)},
        "environment": {"VLLM_NO_USAGE_STATS": "1"},
    }, root=tmp_path)

    runtime.apply_environment()

    assert runtime.layout.results_root == tmp_path / "results"
    assert os.environ["HF_HOME"] == str(tmp_path / ".cache" / "huggingface")
    assert os.environ["MEM0_DIR"] == str(tmp_path / ".cache" / "mem0")
    assert os.environ["VLLM_NO_USAGE_STATS"] == "1"
    assert os.environ["OPENAI_API_KEY"] == "environment-secret"
