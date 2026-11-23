from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

CONTEXT_TERMS = ("context", "prompt", "token", "sequence")
LIMIT_TERMS = ("too long", "maximum", "limit", "length")


@dataclass(slots=True)
class ChatResult:
    content: str
    ttft_ms: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult: ...


class OpenAICompatibleChatClient:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the openai package to use the OpenAI-compatible vLLM client."
            ) from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            request["response_format"] = dict(response_format)
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            if _is_context_limit_error(exc):
                raise RuntimeError(
                    "The vLLM server rejected the prompt as too long. Increase the judge server "
                    "max model length before starting scripts/serve_vllm.sh."
                ) from exc
            raise
        content = _normalize_message_content(response.choices[0].message.content)
        return ChatResult(content=content)


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_part_text(part) for part in content)
    return str(content)


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return _normalize_message_content(part.get("text", part.get("content")))
    text = getattr(part, "text", None)
    if text is None:
        text = getattr(part, "content", None)
    return _normalize_message_content(text)


def _is_context_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(term in message for term in CONTEXT_TERMS) and any(
        term in message for term in LIMIT_TERMS
    )
