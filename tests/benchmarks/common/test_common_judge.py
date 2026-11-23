from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.common.clients import ChatResult, OpenAICompatibleChatClient
from benchmarks.common.judge import (
    failed_judge_payload,
    judge_payload,
    parse_judge_response,
    skipped_judge_payload,
)

MESSAGES = [{"role": "user", "content": "Is the answer correct?"}]


def reply_to(client: OpenAICompatibleChatClient) -> ChatResult:
    return client.chat(MESSAGES, max_tokens=4, temperature=0.0, top_p=1.0)


class FakeOpenAIServer:
    def __init__(self) -> None:
        self.connections: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.reply_content: Any = "true"
        self.failure: BaseException | None = None

    def connect(self, **connection: Any) -> SimpleNamespace:
        self.connections.append(connection)
        completions = SimpleNamespace(create=self.complete)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def complete(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        message = SimpleNamespace(content=self.reply_content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def openai_server(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAIServer:
    server = FakeOpenAIServer()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=server.connect))
    return server


def chat_client() -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        base_url="http://127.0.0.1:8000/v1", api_key="judge-key", model="judge/model"
    )


def test_chat_sends_one_completion_request_to_the_configured_server(openai_server):
    result = chat_client().chat(MESSAGES, max_tokens=4, temperature=0.25, top_p=0.9)

    assert openai_server.connections == [
        {"base_url": "http://127.0.0.1:8000/v1", "api_key": "judge-key"}
    ]
    assert openai_server.requests == [
        {
            "model": "judge/model",
            "messages": MESSAGES,
            "temperature": 0.25,
            "top_p": 0.9,
            "max_tokens": 4,
        }
    ]
    assert result == ChatResult(content="true", ttft_ms=None, metrics={})


def test_only_a_context_length_rejection_is_explained_as_a_prompt_that_is_too_long(openai_server):
    openai_server.failure = ValueError("max_tokens must be positive")
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        reply_to(chat_client())

    openai_server.failure = ValueError("This model's maximum context length is 8192 tokens")
    with pytest.raises(RuntimeError, match="rejected the prompt as too long"):
        reply_to(chat_client())


@pytest.mark.parametrize(
    "response,verdict",
    [
        ("True.", True),
        ("<eos>false</s>", False),
        ('```json\n{"correct": true}\n```', True),
        ('Verdict: {"correct": true} because it matches.', True),
    ],
)
def test_judge_verdict_is_read_from_tokens_json_and_fenced_json(response, verdict):
    assert parse_judge_response(response) == (verdict, "")


@pytest.mark.parametrize(
    "response",
    [
        "the answer is true",
        '{"correct": "true"}',
    ],
)
def test_unreadable_judge_response_becomes_the_reason(response):
    assert parse_judge_response(response) == (None, response)


def test_judge_payloads_record_the_verdict_status_and_metadata_last():
    metadata = {"model": "judge/model", "status": "set by caller"}

    assert judge_payload("true") == {"correct": True, "reason": "", "raw": "true", "status": "ok"}
    assert judge_payload(" maybe ") == {
        "correct": None,
        "reason": "maybe",
        "raw": " maybe ",
        "status": "unparsed",
    }
    assert list(judge_payload("false", metadata).items()) == [
        ("correct", False),
        ("reason", ""),
        ("raw", "false"),
        ("status", "set by caller"),
        ("model", "judge/model"),
    ]
    assert skipped_judge_payload({"model": "judge/model"}) == {
        "correct": None,
        "reason": "skipped",
        "raw": "",
        "status": "skipped",
        "model": "judge/model",
    }
    assert failed_judge_payload(ConnectionError("server down")) == {
        "correct": None,
        "reason": "ConnectionError: server down",
        "raw": "",
        "status": "error",
    }
