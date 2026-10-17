"""Answer text normalization applied before judging and metrics."""

from __future__ import annotations


def final_answer_text(content: str) -> str:
    text = str(content).strip()
    if "ANSWER:" in text:
        return text.rsplit("ANSWER:", 1)[-1].strip()
    return text
