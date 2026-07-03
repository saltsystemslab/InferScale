from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SESSION_RE = re.compile(r"^session_(\d+)$")


@dataclass(slots=True)
class Turn:
    sample_id: str
    session_id: str
    session_index: int
    turn_index: int
    speaker: str
    text: str
    timestamp: str | None = None
    image_caption: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        return f"{self.sample_id}:{self.session_id}:{self.turn_index}"


@dataclass(slots=True)
class SessionChunk:
    sample_id: str
    session_id: str
    session_index: int
    timestamp: str | None
    turns: list[Turn]

    @property
    def id(self) -> str:
        return f"{self.sample_id}:{self.session_id}"


@dataclass(slots=True)
class QuestionAnswer:
    sample_id: str
    question_id: str
    question: str
    answer: str
    category: str
    evidence: Any = None
    raw: dict[str, Any] | None = None


def is_adversarial_category(category: Any) -> bool:
    """LoCoMo category-5 questions are adversarial: unanswerable from the conversation."""
    return str(category).strip().lower() in {"5", "adversarial"}


@dataclass(slots=True)
class ConversationSample:
    sample_id: str
    sessions: list[SessionChunk]
    turns: list[Turn]
    qa: list[QuestionAnswer]
    raw: dict[str, Any]


def load_locomo(path: str | Path, max_samples: int | None = None) -> list[ConversationSample]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        records = data.get("data") or data.get("samples") or data.get("conversations") or []
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported LoCoMo root type: {type(data).__name__}")

    samples: list[ConversationSample] = []
    for index, record in enumerate(records):
        if max_samples is not None and len(samples) >= max_samples:
            break
        if not isinstance(record, dict):
            continue
        sample_id = str(
            record.get("sample_id")
            or record.get("conversation_id")
            or record.get("id")
            or f"sample-{index}"
        )
        turns = list(_extract_turns(sample_id, record))
        sessions = build_session_chunks(sample_id, turns)
        qa = list(_extract_qa(sample_id, record))
        samples.append(ConversationSample(sample_id=sample_id, sessions=sessions, turns=turns, qa=qa, raw=record))
    return samples


def build_session_chunks(sample_id: str, turns: list[Turn]) -> list[SessionChunk]:
    grouped: dict[str, list[Turn]] = {}
    for turn in turns:
        grouped.setdefault(turn.session_id, []).append(turn)

    sessions: list[SessionChunk] = []
    for session_id, session_turns in grouped.items():
        ordered_turns = sorted(session_turns, key=lambda turn: turn.turn_index)
        first = ordered_turns[0]
        sessions.append(
            SessionChunk(
                sample_id=sample_id,
                session_id=session_id,
                session_index=first.session_index,
                timestamp=first.timestamp,
                turns=ordered_turns,
            )
        )
    return sorted(sessions, key=lambda session: session.session_index)


def format_turn_for_memory(turn: Turn) -> str:
    parts = []
    if turn.timestamp:
        parts.append(f"[{turn.timestamp}]")
    if turn.speaker:
        parts.append(f"{turn.speaker}:")
    parts.append(turn.text.strip())
    if turn.image_caption:
        parts.append(f"Image caption: {turn.image_caption.strip()}")
    return " ".join(part for part in parts if part)


def format_session_for_memory(session: SessionChunk) -> str:
    lines = []
    if session.timestamp:
        lines.append(f"[{session.timestamp}]")
    for turn in session.turns:
        parts = []
        if turn.speaker:
            parts.append(f"{turn.speaker}:")
        if turn.text:
            parts.append(turn.text.strip())
        if turn.image_caption:
            parts.append(f"Image caption: {turn.image_caption.strip()}")
        line = " ".join(part for part in parts if part).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _extract_turns(sample_id: str, record: dict[str, Any]) -> Iterable[Turn]:
    conversation = record.get("conversation", record)

    if isinstance(conversation, list):
        yield from _turns_from_session_list(sample_id, "session_1", 1, conversation, None)
        return

    if not isinstance(conversation, dict):
        return

    session_keys: list[tuple[int, str]] = []
    for key in conversation:
        match = SESSION_RE.match(str(key))
        if match and isinstance(conversation.get(key), list):
            session_keys.append((int(match.group(1)), str(key)))

    for session_index, session_key in sorted(session_keys):
        timestamp = _string_or_none(conversation.get(f"{session_key}_date_time"))
        yield from _turns_from_session_list(
            sample_id,
            session_key,
            session_index,
            conversation[session_key],
            timestamp,
        )


def _turns_from_session_list(
    sample_id: str,
    session_id: str,
    session_index: int,
    turns: list[Any],
    session_timestamp: str | None,
) -> Iterable[Turn]:
    for turn_index, turn in enumerate(turns):
        if isinstance(turn, str):
            speaker = ""
            text = turn
            image_caption = None
            raw = {"text": turn}
        elif isinstance(turn, dict):
            speaker = str(turn.get("speaker") or turn.get("role") or "")
            text = str(
                turn.get("text")
                or turn.get("content")
                or turn.get("message")
                or turn.get("utterance")
                or ""
            )
            image_caption = _string_or_none(
                turn.get("blip_caption")
                or turn.get("image_caption")
                or turn.get("caption")
            )
            raw = turn
        else:
            continue
        text = text.strip()
        if not text and not image_caption:
            continue
        yield Turn(
            sample_id=sample_id,
            session_id=session_id,
            session_index=session_index,
            turn_index=turn_index,
            speaker=speaker,
            text=text,
            timestamp=session_timestamp,
            image_caption=image_caption,
            raw=raw,
        )


def _extract_qa(sample_id: str, record: dict[str, Any]) -> Iterable[QuestionAnswer]:
    qa_records = (
        record.get("qa")
        or record.get("qas")
        or record.get("questions")
        or record.get("question_answering")
        or []
    )
    if isinstance(qa_records, dict):
        iterable = qa_records.values()
    else:
        iterable = qa_records
    for index, item in enumerate(iterable):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("query") or "").strip()
        # Category-5 (adversarial) items carry the trap answer in adversarial_answer.
        # Keep that trap in raw, but do not expose it as the gold answer.
        answer = next(
            (
                value
                for key in ("answer", "gold_answer")
                if (value := item.get(key)) is not None
            ),
            "",
        )
        if isinstance(answer, list):
            answer_text = "; ".join(str(part) for part in answer)
        else:
            answer_text = str(answer)
        if not question:
            continue
        question_id = str(item.get("question_id") or item.get("qa_id") or item.get("id") or f"q{index}")
        category = str(item.get("category") or item.get("type") or "unknown")
        yield QuestionAnswer(
            sample_id=sample_id,
            question_id=question_id,
            question=question,
            answer=answer_text,
            category=category,
            evidence=item.get("evidence"),
            raw=item,
        )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
