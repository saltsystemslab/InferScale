from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks.common.clients import ChatResult
from benchmarks.common.files import JsonlWriter, read_jsonl
from benchmarks.common.judge import is_judged, skipped_judge_payload
from benchmarks.rag import judge_run
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.judging import build_rag_judge_messages, judge_rag_record


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
    judge = (
        {"correct": True, "reason": "", "raw": "true", "status": "ok"}
        if judged
        else skipped_judge_payload()
    )
    return {
        "run_id": "r1",
        "mode": "rag-prompt-injection",
        "dataset": "multihoprag",
        "query_id": query_id,
        "question_type": "inference_query",
        "category": "inference_query",
        "question": f"question {query_id}",
        "gold_answer": "gold",
        "gold_answers": ["gold", "golden answer"],
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
        answer_backend="prompt-injection",
        judge_provider="vllm",
        judge_model="stub-judge",
        judge_base_url="http://stub",
        judge_api_key="key",
        rejudge=rejudge,
    )
    config.run_dir.mkdir(parents=True)
    with JsonlWriter(config.run_dir / "predictions.jsonl") as writer:
        writer.write(_record("q0000", judged=False))
        writer.write(_record("q0001", judged=True))
        writer.write(_record("q0002", judged=False))
    (config.run_dir / "config.json").write_text(json.dumps(config.to_jsonable()), encoding="utf-8")
    return config


def test_judge_only_fills_only_missing_records(tmp_path, monkeypatch) -> None:
    config = _write_run(tmp_path)
    stub = StubJudgeClient()
    monkeypatch.setattr(judge_run, "judge_client_for", lambda _: stub)

    summary = judge_run.judge_existing_run(config)

    assert stub.calls == 2
    assert summary["judged_count"] == 3
    assert summary["mode"] == "rag-prompt-injection"
    assert summary["metrics"]["accuracy"] == 1.0
    records = read_jsonl(config.run_dir / "predictions.jsonl")
    assert all(is_judged(record) for record in records)
    assert all(record["mode"] == "rag-prompt-injection" for record in records)
    assert (config.run_dir / "summary.json").exists()
    assert (config.run_dir / "query_metrics.csv").exists()


def test_rejudge_replaces_existing_judgements(tmp_path, monkeypatch) -> None:
    config = _write_run(tmp_path, rejudge=True)
    stub = StubJudgeClient()
    monkeypatch.setattr(judge_run, "judge_client_for", lambda _: stub)

    judge_run.judge_existing_run(config)

    assert stub.calls == 3


def test_judge_failure_persists_progress_and_raises(tmp_path, monkeypatch) -> None:
    config = _write_run(tmp_path)
    stub = StubJudgeClient(fail_at=2)
    monkeypatch.setattr(judge_run, "judge_client_for", lambda _: stub)

    with pytest.raises(RuntimeError, match="rerun the judge stage"):
        judge_run.judge_existing_run(config)

    records = read_jsonl(config.run_dir / "predictions.jsonl")
    by_id = {record["query_id"]: record for record in records}
    assert by_id["q0000"]["judge"]["correct"] is True
    assert by_id["q0002"]["judge"]["status"] == "error"
    assert (config.run_dir / "summary.json").exists()


def test_missing_predictions_file_raises(tmp_path) -> None:
    config = RagBenchConfig(results_dir=tmp_path, run_id="missing")

    with pytest.raises(FileNotFoundError):
        judge_run.judge_existing_run(config)


def test_record_without_references_cannot_be_judged() -> None:
    record = _record("q0000", judged=False)
    del record["gold_answers"]

    with pytest.raises(RuntimeError, match="no gold_answers"):
        judge_rag_record(RagBenchConfig(), StubJudgeClient(), record)


def test_judge_messages_include_question_and_answers() -> None:
    messages = build_rag_judge_messages("Q?", ["gold"], "pred")

    assert len(messages) == 1
    content = messages[0]["content"]
    assert "Question: Q?" in content
    assert "Reference answer: gold" in content
    assert "Predicted answer: pred" in content
    assert "true or false" in content


def test_judge_messages_list_multiple_references() -> None:
    messages = build_rag_judge_messages("Q?", ["first answer", "second answer"], "pred")

    content = messages[0]["content"]
    assert "- first answer" in content
    assert "- second answer" in content
    assert "any one of them" in content
    assert "any one reference answer" in content


def test_judge_messages_accept_a_bare_string() -> None:
    content = build_rag_judge_messages("Q?", "gold", "pred")[0]["content"]

    assert "Reference answer: gold" in content
