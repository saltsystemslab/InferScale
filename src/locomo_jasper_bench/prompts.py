from __future__ import annotations

from .data import ConversationSample, QuestionAnswer
from .vector_types import SearchHit


RETRIEVAL_ANSWER_SYSTEM_PROMPT = (
    "You answer questions about a long conversation. Use the retrieved memory context when it is relevant. "
    "Be concise and do not invent details that are not supported by the context."
)

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating the correctness of an answer about a conversation. "
    "Compare the predicted answer to the reference answer. "
    "Answer with exactly one lowercase word: true or false."
)


def build_retrieval_answer_messages(
    sample: ConversationSample,
    qa: QuestionAnswer,
    hits: list[SearchHit],
) -> list[dict[str, str]]:
    context_lines = []
    for hit in hits:
        memory = hit.payload.get("memory") or hit.payload.get("text") or ""
        context_lines.append(f"{hit.rank}. {memory}")
    context = "\n".join(context_lines) if context_lines else "No retrieved context."
    user = (
        f"Conversation id: {sample.sample_id}\n\n"
        f"Retrieved memory context:\n{context}\n\n"
        f"Question: {qa.question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": RETRIEVAL_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_judge_messages(qa: QuestionAnswer, predicted_answer: str) -> list[dict[str, str]]:
    user = (
        f"Question: {qa.question}\n"
        f"Reference answer: {qa.answer}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        "Return true if the prediction conveys the same essential information as the reference answer, even if worded differently. "
        "Return false for contradictions, unsupported answers, or missing key facts. "
        "Output only true or false. Do not return JSON, punctuation, or an explanation."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
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
