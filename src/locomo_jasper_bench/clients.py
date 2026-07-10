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


OPENAI_JUDGE_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "judge_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "correct": {"type": "boolean"},
            },
            "required": ["correct"],
            "additionalProperties": False,
        },
    },
}


def build_openai_responses_judge_body(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": messages,
        "max_output_tokens": max(32, max_tokens),
        "text": OPENAI_JUDGE_TEXT_FORMAT,
    }


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
        content = _normalize_message_content(response.choices[0].message.content)
        return ChatResult(content=content)


class OpenAIResponsesJudgeClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install openai>=2.44,<3 to use --judge openai.") from exc

            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)

        self._client = client
        self._model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> ChatResult:
        del temperature, top_p
        response = self._client.responses.create(
            **build_openai_responses_judge_body(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
            )
        )
        return ChatResult(content=responses_output_text(response))


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _content_part_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    return str(content)


def _content_part_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        value = item.get("text", item.get("content"))
        return _normalize_message_content(value)
    value = getattr(item, "text", None)
    if value is None:
        value = getattr(item, "content", None)
    return _normalize_message_content(value)


def responses_output_text(response: Any) -> str:
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text
        output = response.get("output")
        if isinstance(output, list):
            return "".join(_normalize_message_content(_dict_or_attr(item, "content")) for item in output)
        return ""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(item, dict):
                content = item.get("content")
            parts.append(_normalize_message_content(content))
        return "".join(parts)
    return ""


def _responses_output_text(response: Any) -> str:
    return responses_output_text(response)


def _dict_or_attr(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _raise_context_limit_error(exc: Exception) -> None:
    text = str(exc).lower()
    context_terms = ("context", "prompt", "token", "sequence")
    limit_terms = ("too long", "maximum", "max", "limit", "length")
    if any(term in text for term in context_terms) and any(term in text for term in limit_terms):
        raise RuntimeError(
            "The vLLM server rejected the prompt as too long. Increase the judge server max model length "
            "before starting scripts/serve_vllm.sh."
        ) from exc
