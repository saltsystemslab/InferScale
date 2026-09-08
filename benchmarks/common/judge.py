"""LLM-as-judge plumbing shared by the benchmarks: payloads, parsing, client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .clients import ChatClient, OpenAICompatibleChatClient


def skipped_judge_payload(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "correct": None,
        "reason": "skipped",
        "raw": "",
        "status": "skipped",
        **dict(metadata or {}),
    }


def failed_judge_payload(exc: Exception, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "correct": None,
        "reason": f"{type(exc).__name__}: {exc}",
        "raw": "",
        "status": "error",
        **dict(metadata or {}),
    }


def judge_payload(content: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    correct, reason = parse_judge_response(content)
    return {
        "correct": correct,
        "reason": reason,
        "raw": content,
        "status": "ok" if isinstance(correct, bool) else "unparsed",
        **dict(metadata or {}),
    }


def is_judged(record: dict[str, Any]) -> bool:
    judge = record.get("judge")
    return isinstance(judge, dict) and isinstance(judge.get("correct"), bool)


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


def build_judge_client(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> ChatClient | None:
    if provider == "none":
        return None
    if provider == "vllm":
        if not base_url:
            raise RuntimeError("A judge base URL is required to use the vllm judge.")
        if not api_key:
            raise RuntimeError("A judge API key is required to use the vllm judge.")
        return OpenAICompatibleChatClient(base_url=base_url, api_key=api_key, model=model)
    raise RuntimeError(f"Unsupported judge provider: {provider}")


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
