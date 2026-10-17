"""LLM-as-judge for LoCoMo answers: the prompt, one verdict per question, and record judging."""

from __future__ import annotations

from typing import Any

from benchmarks.common.clients import ChatClient
from benchmarks.common.judge import judge_with

from .config import MemoryRunConfig
from .data import QuestionAnswer

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating the correctness of an answer about a conversation. "
    "Compare the predicted answer to the reference answer. "
    "Answer with exactly one lowercase word: true or false."
)


def build_judge_messages(
    qa: QuestionAnswer,
    predicted_answer: str,
    *,
    evidence_context: str = "",
) -> list[dict[str, str]]:
    evidence_block = (
        f"Evidence from the conversation:\n{evidence_context}\n\n" if evidence_context else ""
    )
    user = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"{evidence_block}"
        f"Question: {qa.question}\n"
        f"Reference answer: {qa.answer}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        "Return true if the prediction conveys the same essential information as the "
        "reference answer, even if worded differently. "
        "Return false for contradictions, unsupported answers, or missing key facts. "
        "Output only true or false. Do not return JSON, punctuation, or an explanation."
    )
    return [{"role": "user", "content": user}]


def judge_metadata(config: MemoryRunConfig) -> dict[str, Any]:
    return {
        "provider": config.judge_provider,
        "model": config.judge_model,
        "with_evidence": config.with_evidence,
    }


def judge_qa(
    config: MemoryRunConfig,
    judge_client: ChatClient,
    qa: QuestionAnswer,
    predicted_answer: str,
    *,
    evidence_context: str = "",
) -> dict[str, Any]:
    return judge_with(
        judge_client,
        build_judge_messages(qa, predicted_answer, evidence_context=evidence_context),
        max_tokens=config.max_judge_tokens,
        metadata=judge_metadata(config),
    )


def judge_record(
    config: MemoryRunConfig, judge_client: ChatClient, record: dict[str, Any]
) -> dict[str, Any]:
    """Judge a saved prediction record from its own question, gold answer, and evidence."""
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
