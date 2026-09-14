from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.memory import runner
from benchmarks.memory.config import MemoryRunConfig
from memory_config import make_memory_config


class ReachedClientSetup(Exception):
    pass


@pytest.fixture
def config(tmp_path: Path) -> MemoryRunConfig:
    return make_memory_config(
        results_dir=tmp_path,
        run_id="server-provenance",
        vector_backend="qdrant",
        answer_backend="prompt-injection",
    )


@pytest.mark.parametrize(
    ("saved_backend", "saved_settings"),
    [
        ("qdrant", None),
        ("qdrant", {}),
        ("qdrant", {"exact": False}),
        ("qdrant", {"url": "http://another-server:6333"}),
        ("qdrant", {"grpc_port": 7334}),
        ("qdrant", {"prefer_grpc": False}),
        ("qdrant", {"timeout": 120.0}),
        ("jasper", {}),
    ],
)
def test_rejects_qdrant_run_changes_before_touching_artifacts_or_clients(
    config: MemoryRunConfig,
    monkeypatch: pytest.MonkeyPatch,
    saved_backend: str,
    saved_settings: dict[str, Any] | None,
) -> None:
    saved: dict[str, Any] = {"vector_backend": saved_backend}
    if saved_settings is not None:
        saved["qdrant"] = {**config.qdrant.to_dict(), **saved_settings}
    if saved_backend == "qdrant" and saved_settings == {}:
        saved["qdrant"] = {}
    config.run_dir.mkdir(parents=True)
    config_path = config.run_dir / "config.json"
    config_text = json.dumps(saved)
    config_path.write_text(config_text, encoding="utf-8")
    predictions = config.run_dir / "predictions.jsonl"
    predictions.write_text("existing predictions\n", encoding="utf-8")

    def unexpected_clients(*args: Any) -> None:
        pytest.fail("Mismatched Qdrant runs must be rejected before constructing clients.")

    monkeypatch.setattr(runner, "build_clients", unexpected_clients)
    with pytest.raises(RuntimeError, match="Set a new run_id"):
        runner.run_benchmark(config)

    assert config_path.read_text(encoding="utf-8") == config_text
    assert predictions.read_text(encoding="utf-8") == "existing predictions\n"
    assert set(config.run_dir.iterdir()) == {config_path, predictions}


def test_cannot_relabel_existing_qdrant_run_as_jasper(config: MemoryRunConfig) -> None:
    config.run_dir.mkdir(parents=True)
    path = config.run_dir / "config.json"
    saved = config.to_jsonable()
    path.write_text(json.dumps(saved), encoding="utf-8")
    config.vector_backend = "jasper"

    with pytest.raises(RuntimeError, match="Set a new run_id"):
        runner.run_benchmark(config)

    assert json.loads(path.read_text(encoding="utf-8")) == saved


@pytest.mark.parametrize("existing", [False, True], ids=["new-run", "same-settings"])
def test_allows_new_qdrant_runs_and_matching_settings(
    config: MemoryRunConfig, monkeypatch: pytest.MonkeyPatch, existing: bool,
) -> None:
    if existing:
        config.run_dir.mkdir(parents=True)
        (config.run_dir / "config.json").write_text(
            json.dumps(config.to_jsonable()), encoding="utf-8",
        )

    def stop_at_clients(*args: Any) -> None:
        raise ReachedClientSetup

    monkeypatch.setattr(runner, "build_clients", stop_at_clients)
    with pytest.raises(ReachedClientSetup):
        runner.run_benchmark(config)

    recorded = json.loads((config.run_dir / "config.json").read_text(encoding="utf-8"))
    assert recorded["qdrant"] == config.qdrant.to_dict()


def test_jasper_runs_do_not_require_qdrant_provenance(
    config: MemoryRunConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.vector_backend = "jasper"
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "config.json").write_text('{"vector_backend": "jasper"}', encoding="utf-8")

    def stop_at_clients(*args: Any) -> None:
        raise ReachedClientSetup

    monkeypatch.setattr(runner, "build_clients", stop_at_clients)
    with pytest.raises(ReachedClientSetup):
        runner.run_benchmark(config)


def test_deferred_judging_preserves_existing_qdrant_provenance(
    config: MemoryRunConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.run_dir.mkdir(parents=True)
    path = config.run_dir / "config.json"
    saved = {"vector_backend": "qdrant", "with_evidence": False}
    path.write_text(json.dumps(saved), encoding="utf-8")
    (config.run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    summary = {"judged_count": 0, "metrics": {"accuracy": None}}
    monkeypatch.setattr(runner, "judge_client_for", lambda *args: object())
    monkeypatch.setattr(runner, "read_sample_setup_report", lambda *args: {})

    def finish(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["saved_config"] == saved
        return summary

    monkeypatch.setattr(runner, "write_deferred_judging_outputs", finish)
    assert runner.judge_existing_run(config) is summary
    assert json.loads(path.read_text(encoding="utf-8")) == saved
