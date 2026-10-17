from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.memory.config import MemoryRunConfig
from memory_config import make_memory_config, memory_config_data, memory_runtime
from benchmarks.memory.data import ConversationSample, Turn
from benchmarks.memory.mem0 import memory_builder
from benchmarks.memory.mem0.fact_catalog import make_memory_fact
from benchmarks.memory.mem0.memory_builder import fact_catalog_store_for, missing_fact_catalogs
from benchmarks.memory.run import main


def _sample() -> ConversationSample:
    return ConversationSample(
        sample_id="sample-1",
        turns=[
            Turn(
                sample_id="sample-1",
                session_id="session_1",
                session_index=1,
                turn_index=0,
                speaker="Alice",
                text="I like tea.",
                timestamp="2026-01-02",
            )
        ],
        qa=[],
        raw={"conversation": {"speaker_a": "Alice", "speaker_b": "Bob"}},
    )


def _config(tmp_path: Path, **values: object) -> MemoryRunConfig:
    defaults: dict[str, object] = {
        "results_dir": tmp_path / "results",
        "run_id": "check",
        "model": "answer/model",
    }
    defaults.update(values)
    return make_memory_config(
        **defaults,
        runtime=memory_runtime(storage={"cache_root": str(tmp_path / "cache")}),
    )


def _write_catalog(config: MemoryRunConfig, sample: ConversationSample) -> None:
    store = fact_catalog_store_for(config)
    store.write(sample, [make_memory_fact("Alice likes tea.", sample, sample.turns[0])])


def test_missing_fact_catalogs_reports_then_clears(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample = _sample()
    monkeypatch.setattr(memory_builder, "load_locomo", lambda path, max_samples=None: [sample])
    config = _config(tmp_path)

    expected_path = fact_catalog_store_for(config).path_for(sample)
    assert missing_fact_catalogs(config) == [(sample.sample_id, expected_path)]

    _write_catalog(config, sample)
    assert missing_fact_catalogs(config) == []


def test_missing_fact_catalogs_uses_the_full_catalog_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample = _sample()
    monkeypatch.setattr(memory_builder, "load_locomo", lambda path, max_samples=None: [sample])
    _write_catalog(_config(tmp_path), sample)

    other_endpoint = _config(tmp_path, mem0={"llm_base_url": "http://other-host:8000/v1"})
    assert missing_fact_catalogs(other_endpoint)

    other_embedding = _config(tmp_path, inferscale={"embedding": {"model": "other-embedding-model"}})
    assert missing_fact_catalogs(other_embedding)


def test_check_catalogs_fails_with_remediation_and_creates_no_run_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    sample = _sample()
    monkeypatch.setattr(memory_builder, "load_locomo", lambda path, max_samples=None: [sample])
    config_path = tmp_path / "memory.json"
    config_path.write_text(json.dumps(memory_config_data(
        model="answer/model", results_dir=tmp_path / "results", run_id="check", skip_judge=True,
    )), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps({"storage": {"cache_root": str(tmp_path / "cache")}}), encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="1"):
        main(config_path, runtime_path=runtime_path, stage="check-catalogs")

    err = capsys.readouterr().err
    assert "missing Mem0 fact catalogs for model answer/model" in err
    assert "scripts/extract_facts.sh" in err
    assert not (tmp_path / "results" / "check").exists()

    _write_catalog(_config(tmp_path), sample)
    main(config_path, runtime_path=runtime_path, stage="check-catalogs")

    out = capsys.readouterr().out
    assert "fact catalogs complete for model answer/model" in out
    assert not (tmp_path / "results" / "check").exists()
