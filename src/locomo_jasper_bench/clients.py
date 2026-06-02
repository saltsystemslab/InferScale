from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class ChatResult:
    content: str
    latency_ms: float
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    output_tokens_per_sec: float | None = None
    model: str | None = None

    def metrics(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "output_tokens_per_sec": self.output_tokens_per_sec,
            "model": self.model,
        }


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


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        ...


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use vLLM/OpenAI clients.") from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._stream = stream
        self._extra_body = extra_body or {}

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> ChatResult:
        started = time.perf_counter()
        if self._stream:
            return self._chat_streaming(messages, max_tokens, temperature, top_p, started)

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if self._extra_body:
            request["extra_body"] = self._extra_body
        response = self._client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        return ChatResult(
            content=content,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=completion_tokens,
            total_tokens=getattr(usage, "total_tokens", None),
            output_tokens_per_sec=_tokens_per_second(completion_tokens, latency_ms),
            model=getattr(response, "model", self._model),
        )

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
            "stream_options": {"include_usage": True},
        }
        if self._extra_body:
            request["extra_body"] = self._extra_body
        chunks = self._client.chat.completions.create(**request)
        content_parts: list[str] = []
        ttft_ms: float | None = None
        usage = None
        model = self._model
        for chunk in chunks:
            model = getattr(chunk, "model", model)
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            token = getattr(delta, "content", None)
            if token:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
                content_parts.append(token)

        latency_ms = (time.perf_counter() - started) * 1000
        completion_tokens = getattr(usage, "completion_tokens", None)
        return ChatResult(
            content="".join(content_parts),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=completion_tokens,
            total_tokens=getattr(usage, "total_tokens", None),
            output_tokens_per_sec=_tokens_per_second(completion_tokens, latency_ms),
            model=model,
        )


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        batch_size: int = 64,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use OpenAI embeddings.") from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(np.asarray(item.embedding, dtype=np.float32) for item in response.data)
        return vectors


class HashEmbeddingClient:
    """Deterministic local embeddings for tests and dry runs."""

    def __init__(self, dim: int = 1536) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        return vector


def _tokens_per_second(tokens: int | None, latency_ms: float) -> float | None:
    if tokens is None or latency_ms <= 0:
        return None
    return tokens / (latency_ms / 1000)
