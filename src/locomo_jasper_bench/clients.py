from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ChatResult:
    content: str
    ttft_ms: float | None = None


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
        stream: bool = False,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use vLLM/OpenAI clients.") from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._stream = stream

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> ChatResult:
        if self._stream:
            started = time.perf_counter()
            return self._chat_streaming(messages, max_tokens, temperature, top_p, started)

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

    def _chat_streaming(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        started: float,
    ) -> ChatResult:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            chunks = self._client.chat.completions.create(**request)
        except Exception as exc:
            _raise_context_limit_error(exc)
            raise
        content_parts: list[str] = []
        ttft_ms: float | None = None
        try:
            for chunk in chunks:
                if not chunk.choices:
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                token = getattr(delta, "content", None)
                if token:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    content_parts.append(token)
        except Exception as exc:
            _raise_context_limit_error(exc)
            raise

        return ChatResult(content="".join(content_parts), ttft_ms=ttft_ms)


def _raise_context_limit_error(exc: Exception) -> None:
    text = str(exc).lower()
    context_terms = ("context", "prompt", "token", "sequence")
    limit_terms = ("too long", "maximum", "max", "limit", "length")
    if any(term in text for term in context_terms) and any(term in text for term in limit_terms):
        raise RuntimeError(
            "The vLLM server rejected the prompt as too long. Increase VLLM_MAX_MODEL_LEN before starting "
            "scripts/serve_vllm.sh."
        ) from exc
