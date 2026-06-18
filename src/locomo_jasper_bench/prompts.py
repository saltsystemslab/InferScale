from __future__ import annotations

import json
import re
from typing import Any

from .data import ConversationSample, QuestionAnswer
from .vector_types import SearchHit


RETRIEVAL_ANSWER_SYSTEM_PROMPT = (
    "You answer questions about a long conversation. Use the retrieved memory context when it is relevant. "
    "Be concise and do not invent details that are not supported by the context."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for question answering. Compare the predicted answer to the reference answer. "
    "Return exactly one JSON object and nothing else. Use this schema: "
    '{"correct": true, "reason": "short reason"} or {"correct": false, "reason": "short reason"}.'
)


def build_retrieval_answer_messages(
    sample: ConversationSample,
    qa: QuestionAnswer,
    hits: list[SearchHit],
) -> list[dict[str, str]]:
    context_lines = []
    for hit in hits:
        memory = hit.payload.get("memory") or hit.payload.get("text") or ""
        context_lines.append(f"{hit.rank}. {memory}")
    context = "\n".join(context_lines) if context_lines else "No retrieved context."
    user = (
        f"Conversation id: {sample.sample_id}\n\n"
        f"Retrieved memory context:\n{context}\n\n"
        f"Question: {qa.question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": RETRIEVAL_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_judge_messages(qa: QuestionAnswer, predicted_answer: str) -> list[dict[str, str]]:
    user = (
        f"Question: {qa.question}\n"
        f"Reference answer: {qa.answer}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        "Mark correct as true if the prediction captures the same answer, allowing paraphrases and minor wording differences. "
        "Mark correct as false for contradictions, unsupported answers, or missing key facts."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_judge_response(text: str) -> tuple[bool | None, str]:
    stripped = text.strip()
    parsed = _parse_json_object(stripped)
    if isinstance(parsed, dict):
        for key in ("correct", "is_correct", "isCorrect", "answer_correct", "verdict", "judgment", "judgement"):
            if key not in parsed:
                continue
            correct = _coerce_judge_bool(parsed.get(key))
            if correct is not None:
                return correct, str(parsed.get("reason") or parsed.get("explanation") or "")

    field_match = re.search(
        r"\b(?:correct|is_correct|answer_correct|verdict|judg(?:e)?ment)\b\s*[:=]\s*"
        r"(?P<value>true|false|yes|no|correct|incorrect)",
        stripped,
        flags=re.IGNORECASE,
    )
    if field_match:
        correct = _coerce_judge_bool(field_match.group("value"))
        if correct is not None:
            return correct, stripped

    lowered = stripped.lower()
    true_match = re.search(r"\b(true|yes|correct|equivalent)\b|\bnot\s+incorrect\b", lowered)
    false_match = re.search(r"\b(false|no|incorrect|wrong|not\s+correct|not\s+equivalent)\b", lowered)
    if true_match and not false_match:
        return True, stripped
    if false_match and not true_match:
        return False, stripped
    return None, stripped


def _coerce_judge_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower().strip("\"'`.,:; ")
        if lowered in {"true", "yes", "correct", "equivalent", "same", "pass", "1"}:
            return True
        if lowered in {"false", "no", "incorrect", "wrong", "not correct", "not equivalent", "fail", "0"}:
            return False
    return None


def _parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
