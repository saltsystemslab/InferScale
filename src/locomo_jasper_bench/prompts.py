from __future__ import annotations

import json
from typing import Any

from .data import ConversationSample, QuestionAnswer
from .jasper_store import SearchHit


ANSWER_SYSTEM_PROMPT = (
    "You answer questions about a long conversation. Use the retrieved memory context when it is relevant. "
    "Be concise and do not invent details that are not supported by the context."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for question answering. Compare the predicted answer to the reference answer. "
    "Return only a JSON object with keys correct and reason."
)


def build_answer_messages(
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
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
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
        correct = parsed.get("correct")
        if isinstance(correct, bool):
            return correct, str(parsed.get("reason") or "")
        if isinstance(correct, str):
            lowered = correct.lower()
            if lowered in {"true", "yes", "correct"}:
                return True, str(parsed.get("reason") or "")
            if lowered in {"false", "no", "incorrect"}:
                return False, str(parsed.get("reason") or "")
    lowered = stripped.lower()
    if "incorrect" in lowered or '"correct": false' in lowered:
        return False, stripped
    if "correct" in lowered or '"correct": true' in lowered:
        return True, stripped
    return None, stripped


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
