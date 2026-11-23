from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from benchmarks.embedding import model, run
from benchmarks.embedding.corpus import load_corpus

TABLE_HEADER = "batch  facts/s   tokens/s   p50 batch ms   p95 batch ms   amortized ms/fact"
SHUFFLED_FACTS = ["third", "another longer fact", "one fact", "b", "b fact one"]


class NoonInUtc(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> NoonInUtc:
        local_hour, utc_hour = 7, 12
        hour = local_hour if tz is None else utc_hour
        return cls(2026, 11, 23, hour, 0, 0, 123456, tzinfo=tz)


def run_benchmark(monkeypatch: pytest.MonkeyPatch, *arguments: Any) -> None:
    monkeypatch.setattr(sys, "argv", ["run", *map(str, arguments)])
    run.main()


def test_inspect_logs_the_corpus_manifest_without_loading_a_model(
    monkeypatch: pytest.MonkeyPatch, catalog_dir: Path, embedding_log: list[str]
) -> None:
    monkeypatch.setattr(model, "QwenEmbedder", lambda **_: pytest.fail("inspect loaded a model"))
    _, manifest = load_corpus(catalog_dir, seed=3, max_facts=2)

    run_benchmark(
        monkeypatch, "--catalog-dir", catalog_dir, "--inspect", "--seed", 3, "--max-facts", 2
    )

    assert embedding_log == [json.dumps(manifest, indent=2)]
    assert manifest["selected_facts"] == 2


def test_report_is_saved_after_every_batch_size_before_its_row_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    catalog_dir: Path,
    journal: list[tuple[str, Any]],
) -> None:
    run_benchmark(
        monkeypatch,
        "--catalog-dir",
        catalog_dir,
        "--output",
        tmp_path / "report.json",
        *("--batch-sizes", 8, 4),
        *("--warmup", 2),
        *("--repeats", 1),
    )

    model_loaded, *measurements = journal[1:]
    assert journal[0] == ("logged", "Loading Qwen/Qwen3-Embedding-0.6B; corpus contains 5 facts")
    assert model_loaded[0] == "model loaded"
    assert measurements == [
        ("logged", TABLE_HEADER),
        ("encoded", SHUFFLED_FACTS),
        ("encoded", SHUFFLED_FACTS),
        ("encoded", SHUFFLED_FACTS),
        ("report saved", False),
        ("logged", "    8      40.0      120.0        125.000        125.000             25.000"),
        ("encoded", SHUFFLED_FACTS[:4]),
        ("encoded", SHUFFLED_FACTS[4:]),
        ("encoded", SHUFFLED_FACTS[:4]),
        ("encoded", SHUFFLED_FACTS[4:]),
        ("report saved", True),
        ("logged", "    4      20.0       60.0        125.000        125.000             50.000"),
        ("logged", f"Saved {tmp_path / 'report.json'}"),
    ]


def test_report_records_the_environment_the_model_the_corpus_and_every_measurement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    catalog_dir: Path,
    journal: list[tuple[str, Any]],
) -> None:
    monkeypatch.setattr(run, "datetime", NoonInUtc)
    monkeypatch.setattr(run.platform, "python_version", lambda: "3.10.12")
    monkeypatch.setattr(run.platform, "platform", lambda: "Linux-test")
    output = tmp_path / "report.json"
    _, manifest = load_corpus(catalog_dir)

    run_benchmark(
        monkeypatch,
        "--catalog-dir",
        catalog_dir,
        "--output",
        output,
        *("--model", "modèle/embedding"),
        *("--device", "cpu"),
        *("--batch-sizes", 2, 8, 2),
        *("--warmup", 1),
        *("--repeats", 2),
    )

    saved = output.read_text()
    report = json.loads(saved)
    assert saved == json.dumps(report, indent=2) + "\n"
    assert "mod\\u00e8le/embedding" in saved
    measured_with_batches_of_two, measured_with_batches_of_eight = report.pop("results")
    assert list(report.items()) == [
        ("schema_version", 1),
        ("created_at", "2026-11-23T12:00:00.123456+00:00"),
        ("python", "3.10.12"),
        ("platform", "Linux-test"),
        ("model", {"model": "modèle/embedding", "device": "cpu"}),
        (
            "corpus",
            {
                **manifest,
                "token_lengths": {
                    "mean": 3,
                    "p50": 3.0,
                    "p95": 4.0,
                    "p99": 4.0,
                    "min": 2,
                    "max": 4,
                },
            },
        ),
        (
            "timing_scope",
            "tokenization + host/device transfer + forward + pooling + normalization + CPU output",
        ),
        (
            "scheduling",
            "sequential batches, fixed seeded shuffle, no length sorting or embedding cache",
        ),
        ("requested_batch_sizes", [2, 8]),
        ("complete", True),
    ]
    assert list(report["corpus"]) == [*manifest, "token_lengths"]
    assert list(report["corpus"]["token_lengths"]) == ["mean", "p50", "p95", "p99", "min", "max"]
    assert (
        measured_with_batches_of_two["batch_size"],
        measured_with_batches_of_two["measured_batches"],
    ) == (2, 6)
    assert list(measured_with_batches_of_eight.items()) == [
        ("batch_size", 8),
        ("effective_max_batch_size", 5),
        ("warmup_batches", 1),
        ("repeats", 2),
        ("measured_batches", 2),
        ("total_facts", 10),
        ("total_tokens", 30),
        ("embedding_seconds", 0.25),
        ("facts_per_second", 40.0),
        ("tokens_per_second", 120.0),
        ("amortized_ms_per_fact", 25.0),
        ("batch_latency_ms", {"mean": 125.0, "p50": 125.0, "p95": 125.0, "p99": 125.0}),
        ("peak_cuda_allocated_bytes", 4096),
        (
            "batches",
            [
                {"repeat": 0, "start": 0, "facts": 5, "tokens": 15, "latency_ms": 125.0},
                {"repeat": 1, "start": 0, "facts": 5, "tokens": 15, "latency_ms": 125.0},
            ],
        ),
    ]


def test_existing_output_is_refused_before_the_model_loads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    catalog_dir: Path,
    journal: list[tuple[str, Any]],
) -> None:
    output = tmp_path / "report.json"
    output.write_text("kept")

    with pytest.raises(SystemExit) as refused:
        run_benchmark(monkeypatch, "--catalog-dir", catalog_dir, "--output", output)

    assert refused.value.code == 2
    assert f"error: Output already exists: {output}" in capsys.readouterr().err
    assert journal == []
    assert output.read_text() == "kept"
