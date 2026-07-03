from __future__ import annotations

from .data import QuestionAnswer, is_adversarial_category


RETRIEVAL_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that remembers details from past conversations. "
    "Answer questions based on the conversation history provided. "
    "If the answer cannot be determined from the conversation, say "
    "\"I don't have enough information to answer that.\""
)

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating the correctness of an answer about a conversation. "
    "Compare the predicted answer to the reference answer. "
    "Answer with exactly one lowercase word: true or false."
)

JUDGE_ADVERSARIAL_SYSTEM_PROMPT = (
    "You are evaluating how an AI assistant handled a question that cannot be answered "
    "from the conversation it was asked about. "
    "Answer with exactly one lowercase word: true or false."
)


def build_judge_messages(qa: QuestionAnswer, predicted_answer: str) -> list[dict[str, str]]:
    if is_adversarial_category(qa.category):
        return _build_adversarial_judge_messages(qa, predicted_answer)
    user = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Question: {qa.question}\n"
        f"Reference answer: {qa.answer}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        "Return true if the prediction conveys the same essential information as the reference answer, even if worded differently. "
        "Return false for contradictions, unsupported answers, or missing key facts. "
        "Output only true or false. Do not return JSON, punctuation, or an explanation."
    )
    return [
        {"role": "user", "content": user},
    ]


def _build_adversarial_judge_messages(qa: QuestionAnswer, predicted_answer: str) -> list[dict[str, str]]:
    user = (
        f"{JUDGE_ADVERSARIAL_SYSTEM_PROMPT}\n\n"
        f"Question: {qa.question}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        "The question is adversarial: the conversation does not contain the answer, so the correct "
        "behavior is to say the information is not available or to decline to answer. "
        "Return true if the prediction indicates the information is not available, was not mentioned, "
        "or declines to answer. "
        "Return false if the prediction asserts an answer to the question. "
        "Output only true or false. Do not return JSON, punctuation, or an explanation."
    )
    return [
        {"role": "user", "content": user},
    ]


def parse_judge_response(text: str) -> tuple[bool | None, str]:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True, ""
    if lowered == "false":
        return False, ""
    return None, stripped
