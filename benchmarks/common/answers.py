from __future__ import annotations

ANSWER_MARKER = "ANSWER:"


def final_answer_text(content: str) -> str:
    text = str(content).strip()
    if ANSWER_MARKER in text:
        text_after_last_marker = text.rsplit(ANSWER_MARKER, 1)[-1]
        return text_after_last_marker.strip()
    return text
