from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from memory_config import make_memory_config

from benchmarks.common.clients import ChatResult
from benchmarks.memory import runner
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.prediction import PredictionResult
from benchmarks.memory.runtime_clients import RuntimeClients

SYSTEM_METADATA = {"platform": "test"}
SETUP_ROW = {"run_id": "run", "sample_id": "conv-1", "memory_setup_time_ms": 12.5}


def prediction(question_id: str, *, correct: bool | None = None, category: str = "1") -> dict:
    return {
        "run_id": "run",
        "mode": "mem0-kv",
        "sample_id": "conv-1",
        "question_id": question_id,
        "category": category,
        "question": f"Question {question_id}?",
        "gold_answer": "gold",
        "predicted_answer": "predicted",
        "evidence": ["D1:1"],
        "evidence_context": '[D1:1] Alice: "hi"',
        "retrieved_memories": [],
        "judge": {"correct": correct, "status": "skipped" if correct is None else "ok"},
        "metrics": {"kv_memory_tokens": 40, "kv_query_tokens": 10},
    }


class AnswerClient:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class Judge:
    def __init__(self, verdicts: list[Any]) -> None:
        self.verdicts = iter(verdicts)
        self.prompts: list[str] = []

    def chat(self, messages: list[dict[str, str]], **_: Any) -> ChatResult:
        self.prompts.append(messages[0]["content"])
        verdict = next(self.verdicts)
        return ChatResult(verdict)


def run_config(tmp_path: Path, **settings: Any) -> MemoryRunConfig:
    return make_memory_config(results_dir=tmp_path, run_id="run", **settings)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def files_in(folder: Path) -> list[str]:
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file())


@pytest.fixture
def built_client(monkeypatch: pytest.MonkeyPatch) -> AnswerClient:
    client = AnswerClient()
    monkeypatch.setattr(
        runner,
        "build_clients",
        lambda config: RuntimeClients(answer_client=client, judge_client=None),
    )
    monkeypatch.setattr(runner, "collect_system_metadata", lambda: SYSTEM_METADATA)
    return client


def predict(monkeypatch: pytest.MonkeyPatch, records: list[dict], setup_rows: list[dict]) -> None:
    monkeypatch.setattr(
        runner,
        "run_prediction_mode",
        lambda config, clients: PredictionResult(records=records, sample_setup_metrics=setup_rows),
    )


def test_benchmark_writes_the_run_folder_and_returns_its_summary(
    tmp_path, monkeypatch, built_client, memory_log
):
    config = run_config(tmp_path, skip_judge=True)
    predict(monkeypatch, [prediction("q0", correct=True), prediction("q1")], [SETUP_ROW])

    summary = runner.run_benchmark(config)

    assert files_in(config.run_dir) == [
        "config.json",
        "plots/tokens_vs_accuracy_binned.csv",
        "plots/tokens_vs_accuracy_binned.png",
        "plots/tokens_vs_query_to_answer.png",
        "plots/tokens_vs_query_to_first_token.png",
        "plots/tokens_vs_ttft.png",
        "predictions.jsonl",
        "query_metrics.csv",
        "sample_setup_metrics.csv",
        "summary.json",
        "system.json",
    ]
    assert json.loads((config.run_dir / "summary.json").read_text()) == summary
    assert json.loads((config.run_dir / "config.json").read_text()) == config.to_jsonable()
    assert json.loads((config.run_dir / "system.json").read_text()) == SYSTEM_METADATA
    assert [row["question_id"] for row in read_jsonl(config.run_dir / "predictions.jsonl")] == [
        "q0",
        "q1",
    ]
    assert (summary["run_id"], summary["mode"]) == ("run", "mem0-kv")
    assert (summary["question_count"], summary["judged_count"]) == (2, 1)
    assert summary["config"] == config.to_jsonable()
    assert summary["system"] == SYSTEM_METADATA
    assert summary["metrics"]["sample_setup_count"] == 1
    assert memory_log == [
        (
            "INFO",
            f"Starting benchmark run_id=run dataset={config.dataset_path} "
            f"results_dir={config.run_dir}",
        ),
        ("INFO", "Finished benchmark run_id=run questions=2 judged=1 accuracy=1.0000"),
        ("INFO", "Category 1 (multi-hop) accuracy=1.0000 (1/1)"),
        ("INFO", f"Wrote results to {config.run_dir}"),
    ]


def test_benchmark_closes_the_answer_client_it_built_when_prediction_fails(
    tmp_path, monkeypatch, built_client
):
    def run_prediction_mode(config, clients):
        raise RuntimeError("engine died")

    monkeypatch.setattr(runner, "run_prediction_mode", run_prediction_mode)
    config = run_config(tmp_path, skip_judge=True)

    with pytest.raises(RuntimeError, match="engine died"):
        runner.run_benchmark(config)

    assert built_client.close_count == 1
    assert files_in(config.run_dir) == ["config.json", "system.json"]


@pytest.fixture
def saved_run(monkeypatch: pytest.MonkeyPatch):
    def save(config: MemoryRunConfig, records: list[dict], judge: Judge) -> Judge:
        config.run_dir.mkdir(parents=True)
        (config.run_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        monkeypatch.setattr(runner, "judge_client_for", lambda config: judge)
        return judge

    return save


def test_judging_fills_in_only_the_unjudged_records(tmp_path, saved_run, memory_log):
    config = run_config(tmp_path)
    records = [prediction("q0", correct=True), prediction("q1"), prediction("q2", category="2")]
    judge = saved_run(config, records, Judge(["false", "true"]))

    summary = runner.judge_existing_run(config)

    assert ["Question q1?" in prompt for prompt in judge.prompts] == [True, False]
    assert ["Question q2?" in prompt for prompt in judge.prompts] == [False, True]
    judged = read_jsonl(config.run_dir / "predictions.jsonl")
    assert [record["judge"]["correct"] for record in judged] == [True, False, True]
    assert (summary["judged_count"], summary["correct_count"]) == (3, 2)
    assert json.loads((config.run_dir / "summary.json").read_text()) == summary
    assert memory_log == [
        (
            "INFO",
            f"Judging existing run_id=run predictions={config.run_dir / 'predictions.jsonl'}",
        ),
        ("INFO", "Finished deferred judging run_id=run judged_now=2 judged=3 accuracy=0.6667"),
        ("INFO", "Category 1 (multi-hop) accuracy=0.5000 (1/2)"),
        ("INFO", "Category 2 (temporal) accuracy=1.0000 (1/1)"),
    ]


def test_judging_keeps_the_config_system_and_setup_metrics_saved_with_the_run(tmp_path, saved_run):
    config = run_config(tmp_path)
    saved_run(config, [prediction("q0")], Judge(["true"]))
    (config.run_dir / "config.json").write_text(json.dumps({"saved": "config"}))
    (config.run_dir / "system.json").write_text(json.dumps({"saved": "system"}))
    (config.run_dir / "sample_setup_metrics.csv").write_text(
        "sample_id,memory_setup_time_ms\nconv-1,12.5\nconv-2,7.5\n"
    )

    summary = runner.judge_existing_run(config)

    assert summary["config"] == {"saved": "config"}
    assert summary["system"] == {"saved": "system"}
    assert summary["metrics"]["sample_setup_count"] == 2
    assert summary["metrics"]["memory_setup_time_ms"]["avg"] == 10.0
