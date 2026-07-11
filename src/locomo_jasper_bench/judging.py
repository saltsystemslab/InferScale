from __future__ import annotations

from typing import Any

from .clients import ChatClient
from .config import BenchmarkConfig
from .data import QuestionAnswer
from .prompts import build_judge_messages, parse_judge_response


def skipped_judge_payload(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    return {
        "correct": None,
        "reason": "skipped",
        "raw": "",
        "status": "skipped",
        **_judge_metadata(config),
    }


def failed_judge_payload(exc: Exception, config: BenchmarkConfig | None = None) -> dict[str, Any]:
    return {
        "correct": None,
        "reason": f"{type(exc).__name__}: {exc}",
        "raw": "",
        "status": "error",
        **_judge_metadata(config),
    }


def judge_qa(
    config: BenchmarkConfig,
    judge_client: ChatClient,
    qa: QuestionAnswer,
    predicted_answer: str,
) -> dict[str, Any]:
    judge_messages = build_judge_messages(qa, predicted_answer)
    judge = judge_client.chat(
        judge_messages,
        max_tokens=config.max_judge_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return parsed_judge_payload(config, judge.content)


def parsed_judge_payload(config: BenchmarkConfig, content: str) -> dict[str, Any]:
    correct, reason = parse_judge_response(content)
    return {
        "correct": correct,
        "reason": reason,
        "raw": content,
        "status": "ok" if isinstance(correct, bool) else "unparsed",
        **_judge_metadata(config),
    }


def judge_record(config: BenchmarkConfig, judge_client: ChatClient, record: dict[str, Any]) -> dict[str, Any]:
    qa = QuestionAnswer(
        sample_id=str(record.get("sample_id") or ""),
        question_id=str(record.get("question_id") or ""),
        question=str(record.get("question") or ""),
        answer=str(record.get("gold_answer") or ""),
        category=str(record.get("category") or ""),
        evidence=record.get("evidence"),
    )
    return judge_qa(config, judge_client, qa, str(record.get("predicted_answer") or ""))


def is_judged(record: dict[str, Any]) -> bool:
    judge = record.get("judge")
    return isinstance(judge, dict) and isinstance(judge.get("correct"), bool)


def record_label(record: dict[str, Any]) -> str:
    return (
        f"sample_id={record.get('sample_id') or ''} "
        f"question_id={record.get('question_id') or ''} "
        f"category={record.get('category') or ''}"
    ).strip()


def judge_label(value: Any) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "incorrect"
    return "skipped"


def format_accuracy(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _judge_metadata(config: BenchmarkConfig | None) -> dict[str, Any]:
    if config is None:
        return {}
    return {
        "provider": config.judge_provider,
        "model": config.judge_model,
        "with_evidence": config.with_evidence,
    }
