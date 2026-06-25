from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


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
    ) -> ChatResult:
        ...


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use vLLM/OpenAI clients.") from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> ChatResult:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            _raise_context_limit_error(exc)
            raise
        content = response.choices[0].message.content or ""
        return ChatResult(content=content)


def _raise_context_limit_error(exc: Exception) -> None:
    text = str(exc).lower()
    context_terms = ("context", "prompt", "token", "sequence")
    limit_terms = ("too long", "maximum", "max", "limit", "length")
    if any(term in text for term in context_terms) and any(term in text for term in limit_terms):
        raise RuntimeError(
            "The vLLM server rejected the prompt as too long. Increase LOCOMO_VLLM_MAX_MODEL_LEN before "
            "starting scripts/serve_vllm.sh."
        ) from exc
