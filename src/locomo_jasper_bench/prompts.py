from __future__ import annotations

import json

from .data import QuestionAnswer


JUDGE_SYSTEM_PROMPT = (
    "You are evaluating the correctness of an answer about a conversation. "
    "Compare the predicted answer to the reference answer. "
    "Answer with exactly one lowercase word: true or false."
)

def build_judge_messages(
    qa: QuestionAnswer,
    predicted_answer: str,
) -> list[dict[str, str]]:
    output_instruction = (
        "Return true if the prediction conveys the same essential information as the reference answer, even if worded differently. "
        "Return false for contradictions, unsupported answers, or missing key facts. "
        "Output only true or false. Do not return JSON, punctuation, or an explanation."
    )
    user = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Question: {qa.question}\n"
        f"Reference answer: {qa.answer}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        f"{output_instruction}"
    )
    return [{"role": "user", "content": user}]


def parse_judge_response(text: str) -> tuple[bool | None, str]:
    stripped = text.strip()
    parsed = _parse_json_verdict(stripped)
    if parsed is not None:
        return parsed, ""
    parsed = _parse_boolean_token(stripped)
    if parsed is not None:
        return parsed, ""
    return None, stripped


def _parse_json_verdict(text: str) -> bool | None:
    candidates = [text]
    fenced = _strip_code_fence(text)
    if fenced != text:
        candidates.append(fenced)
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        parsed = _json_value_to_bool(value)
        if parsed is not None:
            return parsed
    return None


def _json_value_to_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_boolean_token(value)
    if isinstance(value, dict):
        correct = value.get("correct")
        return correct if isinstance(correct, bool) else None
    return None


def _parse_boolean_token(text: str) -> bool | None:
    cleaned = _strip_code_fence(_strip_known_model_tokens(text)).strip()
    token = cleaned.strip(" \t\r\n`'\".,;:!()[]{}")
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _strip_known_model_tokens(text: str) -> str:
    cleaned = text
    for token in (
        "<end_of_turn>",
        "<eos>",
        "</s>",
        "<|eot_id|>",
        "<|endoftext|>",
    ):
        cleaned = cleaned.replace(token, " ")
    return cleaned
