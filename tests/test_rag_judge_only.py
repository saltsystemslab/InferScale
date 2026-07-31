from __future__ import annotations

import json
from typing import Any

import pytest

from locomo_jasper_bench.clients import ChatResult
from locomo_jasper_bench.results import JsonlWriter
from locomo_jasper_bench.run_files import read_jsonl
from rag_bench import runner as rag_runner
from rag_bench.config import RagBenchConfig
from rag_bench.judging import (
    build_rag_judge_messages,
    is_judged,
    skipped_judge_payload,
)


class StubJudgeClient:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        del messages, max_tokens, temperature, top_p, response_format
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("judge server down")
        return ChatResult(content="true")


def _record(query_id: str, *, judged: bool) -> dict[str, Any]:
    judge = {"correct": True, "reason": "", "raw": "true", "status": "ok"} if judged else (
        skipped_judge_payload()
    )
    return {
        "run_id": "r1",
        "mode": "rag-prefix",
        "dataset": "multihoprag",
        "query_id": query_id,
        "question_type": "inference_query",
        "category": "inference_query",
        "question": f"question {query_id}",
        "gold_answer": "gold",
        "predicted_answer": "gold",
        "evidence": [],
        "retrieved_chunks": [],
        "answer_metrics": {
            "exact_match": True,
            "f1": 1.0,
            "substring_match": True,
            "predicted_insufficient": False,
        },
        "retrieval": None,
        "judge": judge,
        "metrics": {},
    }


def _write_run(tmp_path, *, rejudge: bool = False) -> RagBenchConfig:
    config = RagBenchConfig(
        results_dir=tmp_path,
        run_id="r1",
        answer_backend="vllm-prefix",
        judge_provider="vllm",
        judge_model="stub-judge",
        judge_base_url="http://stub",
        judge_api_key="key",
        judge_only=True,
        rejudge=rejudge,
    )
    config.run_dir.mkdir(parents=True)
    with JsonlWriter(config.run_dir / "predictions.jsonl") as writer:
        writer.write(_record("q0000", judged=False))
        writer.write(_record("q0001", judged=True))
        writer.write(_record("q0002", judged=False))
    (config.run_dir / "config.json").write_text(
        json.dumps(config.to_jsonable()), encoding="utf-8"
    )
    return config


def test_judge_only_fills_only_missing_records(tmp_path, monkeypatch) -> None:
    config = _write_run(tmp_path)
    stub = StubJudgeClient()
    monkeypatch.setattr(rag_runner, "build_judge_client", lambda _: stub)

    summary = rag_runner.judge_existing_run(config)

    assert stub.calls == 2
    assert summary["judged_count"] == 3
    assert summary["mode"] == "rag-prefix"
    assert summary["metrics"]["accuracy"] == 1.0
    records = read_jsonl(config.run_dir / "predictions.jsonl")
    assert all(is_judged(record) for record in records)
    assert (config.run_dir / "summary.json").exists()
    assert (config.run_dir / "query_metrics.csv").exists()


def test_rejudge_replaces_existing_judgements(tmp_path, monkeypatch) -> None:
    config = _write_run(tmp_path, rejudge=True)
    stub = StubJudgeClient()
    monkeypatch.setattr(rag_runner, "build_judge_client", lambda _: stub)

    rag_runner.judge_existing_run(config)

    assert stub.calls == 3


def test_judge_failure_persists_progress_and_raises(tmp_path, monkeypatch) -> None:
    config = _write_run(tmp_path)
    stub = StubJudgeClient(fail_at=2)
    monkeypatch.setattr(rag_runner, "build_judge_client", lambda _: stub)

    with pytest.raises(RuntimeError, match="rerun --judge-only"):
        rag_runner.judge_existing_run(config)

    records = read_jsonl(config.run_dir / "predictions.jsonl")
    by_id = {record["query_id"]: record for record in records}
    assert by_id["q0000"]["judge"]["correct"] is True
    assert by_id["q0002"]["judge"]["status"] == "error"
    assert (config.run_dir / "summary.json").exists()


def test_missing_predictions_file_raises(tmp_path) -> None:
    config = RagBenchConfig(results_dir=tmp_path, run_id="missing", judge_only=True)

    with pytest.raises(FileNotFoundError):
        rag_runner.judge_existing_run(config)


def test_judge_messages_include_question_and_answers() -> None:
    messages = build_rag_judge_messages("Q?", "gold", "pred")

    assert len(messages) == 1
    content = messages[0]["content"]
    assert "Question: Q?" in content
    assert "Reference answer: gold" in content
    assert "Predicted answer: pred" in content
    assert "true or false" in content
