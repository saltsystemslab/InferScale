from __future__ import annotations

from typing import Any

from benchmarks.common.clients import ChatResult
from benchmarks.memory.data import QuestionAnswer
from memory_config import make_memory_config
from benchmarks.memory.judging import judge_qa
from benchmarks.memory.judge_prompt import build_judge_messages
from benchmarks.common.judge import parse_judge_response


def _qa() -> QuestionAnswer:
    return QuestionAnswer(
        sample_id="sample",
        question_id="question",
        question="Where did they travel?",
        answer="Paris",
        category="1",
    )


def test_build_judge_messages_uses_simple_boolean_protocol() -> None:
    messages = build_judge_messages(_qa(), "They traveled to Paris.")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Reference answer: Paris" in messages[0]["content"]
    assert "Output only true or false" in messages[0]["content"]
    assert "JSON" in messages[0]["content"]


def test_parse_boolean_and_compatible_json_verdicts() -> None:
    assert parse_judge_response("true") == (True, "")
    assert parse_judge_response("false") == (False, "")
    assert parse_judge_response('{"correct": true}') == (True, "")
    assert parse_judge_response("not parseable") == (None, "not parseable")


def test_judge_does_not_request_structured_output() -> None:
    class RecordingJudge:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def chat(self, _messages: Any, **kwargs: Any) -> ChatResult:
            self.kwargs = kwargs
            return ChatResult("true")

    judge = RecordingJudge()
    payload = judge_qa(make_memory_config(), judge, _qa(), "Paris")  # type: ignore[arg-type]

    assert payload["correct"] is True
    assert "response_format" not in judge.kwargs


def test_with_evidence_setting_remains_available_for_saved_run_compatibility() -> None:
    assert make_memory_config(skip_judge=True).with_evidence is False
    assert make_memory_config(skip_judge=True, judge={'with_evidence': True}).with_evidence is True
