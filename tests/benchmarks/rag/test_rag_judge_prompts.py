from __future__ import annotations

from typing import Any

import pytest

from benchmarks.common.clients import ChatResult
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.judging import build_rag_judge_messages, judge_metadata, judge_rag_answer

INSTRUCTIONS = (
    "You are evaluating the correctness of an answer to a question about a collection "
    "of documents. Compare the predicted answer to the reference answers. "
    "Answer with exactly one lowercase word: true or false."
)
VERDICT_RULES = (
    "even if worded differently. "
    "Return false for contradictions, unsupported answers, or missing key facts. "
    "Output only true or false. Do not return JSON, punctuation, or an explanation."
)


class RecordingJudge:
    def __init__(self, verdict: str = "true") -> None:
        self.verdict = verdict
        self.requests: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, str]], **options: Any) -> ChatResult:
        self.requests.append((messages, options))
        return ChatResult(content=self.verdict)


@pytest.mark.parametrize("gold_answers", [["  Alpha Corp  "]])
def test_single_reference_prompt_is_byte_for_byte_stable(gold_answers):
    assert build_rag_judge_messages("Who won?", gold_answers, "Alpha") == [
        {
            "role": "user",
            "content": (
                f"{INSTRUCTIONS}\n\n"
                "Question: Who won?\n"
                "Reference answer: Alpha Corp\n"
                "Predicted answer: Alpha\n\n"
                "Return true if the prediction conveys the same essential information as "
                f"the reference answer, {VERDICT_RULES}"
            ),
        }
    ]


def test_several_references_prompt_is_byte_for_byte_stable():
    assert build_rag_judge_messages("Who won?", ["Alpha Corp", "", 42], "Alpha") == [
        {
            "role": "user",
            "content": (
                f"{INSTRUCTIONS}\n\n"
                "Question: Who won?\n"
                "Reference answers (independent annotations; the prediction is correct "
                "if it matches any one of them):\n"
                "- Alpha Corp\n"
                "- 42\n"
                "Predicted answer: Alpha\n\n"
                "Return true if the prediction conveys the same essential information as "
                f"any one reference answer, {VERDICT_RULES}"
            ),
        }
    ]


def test_judge_is_asked_within_the_configured_token_limit_and_tagged_with_its_model():
    config = RagBenchConfig(judge_model="judge/model", max_judge_tokens=9)
    judge = RecordingJudge(verdict="false")

    payload = judge_rag_answer(
        config, judge, question="Who won?", gold_answers=("Alpha Corp",), predicted_answer="Beta"
    )

    assert payload == {
        "correct": False,
        "reason": "",
        "raw": "false",
        "status": "ok",
        "provider": "vllm",
        "model": "judge/model",
    }
    assert judge.requests == [
        (
            build_rag_judge_messages("Who won?", ("Alpha Corp",), "Beta"),
            {"max_tokens": 9, "temperature": 0.0, "top_p": 1.0},
        )
    ]
    assert judge_metadata(config) == {"provider": "vllm", "model": "judge/model"}
