from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.memory.config import MemoryRunConfig
from memory_config import make_memory_config
from benchmarks.memory.runner import _reconcile_with_evidence


def _config(tmp_path: Path, *, with_evidence: bool, rejudge: bool) -> MemoryRunConfig:
    return make_memory_config(results_dir=tmp_path, run_id='run', rejudge=rejudge, judge={'with_evidence': with_evidence})


def test_judge_only_adopts_the_saved_evidence_protocol(tmp_path: Path) -> None:
    config = _config(tmp_path, with_evidence=False, rejudge=False)

    saved = _reconcile_with_evidence(config, {"with_evidence": True})

    assert config.with_evidence is True
    assert saved["with_evidence"] is True


def test_judge_only_defaults_old_runs_to_no_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path, with_evidence=False, rejudge=False)

    _reconcile_with_evidence(config, {})

    assert config.with_evidence is False


def test_judge_only_rejects_explicit_evidence_mismatch_without_rejudge(tmp_path: Path) -> None:
    config = _config(tmp_path, with_evidence=True, rejudge=False)

    with pytest.raises(RuntimeError, match="pass --rejudge"):
        _reconcile_with_evidence(config, {"with_evidence": False})


def test_rejudge_overrides_and_rewrites_the_saved_config(tmp_path: Path) -> None:
    config = _config(tmp_path, with_evidence=True, rejudge=True)
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "config.json").write_text(
        json.dumps({"with_evidence": False}) + "\n",
        encoding="utf-8",
    )

    saved = _reconcile_with_evidence(config, {"with_evidence": False})

    assert config.with_evidence is True
    assert saved["with_evidence"] is True
    rewritten = json.loads((config.run_dir / "config.json").read_text(encoding="utf-8"))
    assert rewritten["with_evidence"] is True
