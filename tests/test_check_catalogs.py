from __future__ import annotations

from pathlib import Path

import pytest

from locomo_jasper_bench.config import BenchmarkConfig
from locomo_jasper_bench.data import ConversationSample, Turn
from locomo_jasper_bench.retrieval import memory_builder
from locomo_jasper_bench.retrieval.fact_catalog import make_memory_fact
from locomo_jasper_bench.retrieval.memory_builder import (
    fact_catalog_store_for,
    missing_fact_catalogs,
)
from locomo_jasper_bench.run import main


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


def _config(tmp_path: Path, **overrides: object) -> BenchmarkConfig:
    defaults: dict[str, object] = {
        "results_dir": tmp_path / "results",
        "run_id": "check",
        "model": "answer/model",
        "memory_llm_cache_dir": tmp_path / "mem0-inference",
        "embedding_cache_dir": tmp_path / "embedding-cache",
    }
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)  # type: ignore[arg-type]


def _write_catalog(config: BenchmarkConfig, sample: ConversationSample) -> None:
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

    other_endpoint = _config(tmp_path, memory_llm_base_url="http://other-host:9000/v1")
    assert missing_fact_catalogs(other_endpoint)

    other_embedding = _config(tmp_path, embedding_model="other-embedding-model")
    assert missing_fact_catalogs(other_embedding)


def test_check_catalogs_cli_fails_with_remediation_and_creates_no_run_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample = _sample()
    monkeypatch.setattr(memory_builder, "load_locomo", lambda path, max_samples=None: [sample])
    argv = [
        "--check-catalogs",
        "--skip-judge",
        "--answer-model",
        "answer/model",
        "--results-dir",
        str(tmp_path / "results"),
        "--run-id",
        "check",
        "--memory-llm-cache-dir",
        str(tmp_path / "mem0-inference"),
        "--embedding-cache-dir",
        str(tmp_path / "embedding-cache"),
    ]

    with pytest.raises(SystemExit, match="1"):
        main(argv)

    err = capsys.readouterr().err
    assert "missing Mem0 fact catalogs for model answer/model" in err
    assert 'EXTRACTION_MODELS="answer/model" bash scripts/extract_facts.sh' in err
    assert not (tmp_path / "results" / "check").exists()

    _write_catalog(_config(tmp_path), sample)
    main(argv)

    out = capsys.readouterr().out
    assert "fact catalogs complete for model answer/model" in out
    assert not (tmp_path / "results" / "check").exists()
