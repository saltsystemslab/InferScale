from __future__ import annotations

from typing import Any

from benchmarks.common.clients import ChatClient
from benchmarks.common.judge import judge_payload

from .config import BenchmarkConfig
from .data import QuestionAnswer
from .judge_prompt import build_judge_messages


def judge_metadata(config: BenchmarkConfig | None) -> dict[str, Any]:
    if config is None:
        return {}
    return {
        "provider": config.judge_provider,
        "model": config.judge_model,
        "with_evidence": config.with_evidence,
    }


def judge_qa(
    config: BenchmarkConfig,
    judge_client: ChatClient,
    qa: QuestionAnswer,
    predicted_answer: str,
    *,
    evidence_context: str = "",
) -> dict[str, Any]:
    judge_messages = build_judge_messages(qa, predicted_answer, evidence_context=evidence_context)
    judge = judge_client.chat(
        judge_messages,
        max_tokens=config.max_judge_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return judge_payload(judge.content, judge_metadata(config))


def judge_record(config: BenchmarkConfig, judge_client: ChatClient, record: dict[str, Any]) -> dict[str, Any]:
    qa = QuestionAnswer(
        sample_id=str(record.get("sample_id") or ""),
        question_id=str(record.get("question_id") or ""),
        question=str(record.get("question") or ""),
        answer=str(record.get("gold_answer") or ""),
        category=str(record.get("category") or ""),
        evidence=record.get("evidence"),
    )
    evidence_context = str(record.get("evidence_context") or "") if config.with_evidence else ""
    return judge_qa(
        config,
        judge_client,
        qa,
        str(record.get("predicted_answer") or ""),
        evidence_context=evidence_context,
    )


def record_label(record: dict[str, Any]) -> str:
    return (
        f"sample_id={record.get('sample_id') or ''} "
        f"question_id={record.get('question_id') or ''} "
        f"category={record.get('category') or ''}"
    ).strip()
