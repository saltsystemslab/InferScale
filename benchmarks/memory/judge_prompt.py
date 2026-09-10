from __future__ import annotations

from .data import QuestionAnswer


JUDGE_SYSTEM_PROMPT = (
    "You are evaluating the correctness of an answer about a conversation. "
    "Compare the predicted answer to the reference answer. "
    "Answer with exactly one lowercase word: true or false."
)


def build_judge_messages(
    qa: QuestionAnswer,
    predicted_answer: str,
    *,
    evidence_context: str = "",
) -> list[dict[str, str]]:
    evidence_block = f"Evidence from the conversation:\n{evidence_context}\n\n" if evidence_context else ""
    user = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"{evidence_block}"
        f"Question: {qa.question}\n"
        f"Reference answer: {qa.answer}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        "Return true if the prediction conveys the same essential information as the reference answer, even if worded differently. "
        "Return false for contradictions, unsupported answers, or missing key facts. "
        "Output only true or false. Do not return JSON, punctuation, or an explanation."
    )
    return [{"role": "user", "content": user}]
