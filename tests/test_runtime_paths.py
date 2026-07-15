from __future__ import annotations

from pathlib import Path

import pytest

from locomo_jasper_bench.runtime_paths import local_store_scratch_dir


def test_local_store_scratch_dir_defaults_to_container_local_tmp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCOMO_LOCAL_STORE_DIR", raising=False)
    # TMPDIR must not influence the result: remote scratch env points it at the
    # network volume, which cannot host sqlite.
    monkeypatch.setenv("TMPDIR", "/workspace/tmp")

    assert local_store_scratch_dir("run-1") == Path("/tmp/locomo-jasper-stores/run-1")


def test_local_store_scratch_dir_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCOMO_LOCAL_STORE_DIR", "/mnt/nvme")

    assert local_store_scratch_dir("run-1") == Path("/mnt/nvme/locomo-jasper-stores/run-1")


def test_local_store_scratch_dir_separates_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCOMO_LOCAL_STORE_DIR", raising=False)

    assert local_store_scratch_dir("run-a") != local_store_scratch_dir("run-b")
