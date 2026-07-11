from __future__ import annotations

from typing import Any

from locomo_jasper_bench.clients import ChatResult
from locomo_jasper_bench.config import BenchmarkConfig, parse_args
from locomo_jasper_bench.data import QuestionAnswer
from locomo_jasper_bench.judging import judge_qa
from locomo_jasper_bench.prompts import build_judge_messages, parse_judge_response


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
    payload = judge_qa(BenchmarkConfig(), judge, _qa(), "Paris")  # type: ignore[arg-type]

    assert payload["correct"] is True
    assert "response_format" not in judge.kwargs


def test_with_evidence_flag_remains_parseable_for_saved_run_compatibility() -> None:
    assert parse_args(["--skip-judge"]).with_evidence is False
    assert parse_args(["--skip-judge", "--with-evidence"]).with_evidence is True
