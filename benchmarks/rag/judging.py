from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from benchmarks.common.clients import ChatClient
from benchmarks.common.judge import build_judge_client, judge_payload

from .config import RagBenchConfig

RAG_JUDGE_INSTRUCTIONS = (
    "You are evaluating the correctness of an answer to a question about a collection "
    "of documents. Compare the predicted answer to the reference answers. "
    "Answer with exactly one lowercase word: true or false."
)


def build_rag_judge_messages(
    question: str,
    gold_answers: Sequence[str] | str,
    predicted_answer: str,
) -> list[dict[str, str]]:
    """Judge prompt over one or more reference answers.

    Datasets with several independent annotations (QASPER) list every
    reference; the prediction is correct if it matches any one of them.
    """
    if isinstance(gold_answers, str):
        gold_answers = (gold_answers,)
    references = [str(gold).strip() for gold in gold_answers if str(gold).strip()] or [""]
    if len(references) == 1:
        reference_block = f"Reference answer: {references[0]}\n"
        match_clause = "the reference answer"
    else:
        listed = "\n".join(f"- {reference}" for reference in references)
        reference_block = (
            "Reference answers (independent annotations; the prediction is correct "
            f"if it matches any one of them):\n{listed}\n"
        )
        match_clause = "any one reference answer"
    user = (
        f"{RAG_JUDGE_INSTRUCTIONS}\n\n"
        f"Question: {question}\n"
        f"{reference_block}"
        f"Predicted answer: {predicted_answer}\n\n"
        f"Return true if the prediction conveys the same essential information as "
        f"{match_clause}, even if worded differently. "
        "Return false for contradictions, unsupported answers, or missing key facts. "
        "Output only true or false. Do not return JSON, punctuation, or an explanation."
    )
    return [{"role": "user", "content": user}]


def judge_metadata(config: RagBenchConfig | None) -> dict[str, Any]:
    if config is None:
        return {}
    return {
        "provider": config.judge_provider,
        "model": config.judge_model,
    }


def judge_client_for(config: RagBenchConfig) -> ChatClient | None:
    if config.skip_judge or config.judge_provider != "vllm":
        return None
    return build_judge_client(
        provider=config.judge_provider,
        model=config.judge_model,
        base_url=config.judge_base_url,
        api_key=config.judge_api_key,
    )


def judge_rag_answer(
    config: RagBenchConfig,
    judge_client: ChatClient,
    *,
    question: str,
    gold_answers: Sequence[str] | str,
    predicted_answer: str,
) -> dict[str, Any]:
    judge = judge_client.chat(
        build_rag_judge_messages(question, gold_answers, predicted_answer),
        max_tokens=config.max_judge_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return judge_payload(judge.content, judge_metadata(config))


def judge_rag_record(
    config: RagBenchConfig,
    judge_client: ChatClient,
    record: dict[str, Any],
) -> dict[str, Any]:
    gold_answers = record.get("gold_answers")
    if not isinstance(gold_answers, list) or not gold_answers:
        gold_answers = [str(record.get("gold_answer") or "")]
    return judge_rag_answer(
        config,
        judge_client,
        question=str(record.get("question") or ""),
        gold_answers=[str(gold) for gold in gold_answers],
        predicted_answer=str(record.get("predicted_answer") or ""),
    )


def record_label(record: dict[str, Any]) -> str:
    return (
        f"query_id={record.get('query_id') or ''} "
        f"question_type={record.get('question_type') or ''}"
    ).strip()
