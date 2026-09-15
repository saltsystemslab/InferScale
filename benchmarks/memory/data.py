from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
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
class QuestionAnswer:
    sample_id: str
    question_id: str
    question: str
    answer: str
    category: str
    evidence: Any = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class ConversationSample:
    sample_id: str
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
        qa = list(_extract_qa(sample_id, record))
        samples.append(ConversationSample(sample_id=sample_id, turns=turns, qa=qa, raw=record))
    return samples


def format_turn_for_memory(turn: Turn) -> str:
    text = turn.text.strip()
    raw = turn.raw if isinstance(turn.raw, dict) else {}
    image_query = str(raw.get("query") or "").strip()
    image_caption = str(turn.image_caption or "").strip()
    if image_query and image_caption:
        photo_tag = f"[Sharing image - query: {image_query}. The image shows: {image_caption}]"
    elif image_query:
        photo_tag = f"[Sharing image - query for: {image_query}]"
    elif image_caption:
        photo_tag = f"[Sharing image that shows: {image_caption}]"
    else:
        photo_tag = ""
    content = " ".join(part for part in (text, photo_tag) if part)
    if turn.speaker:
        return f"{turn.speaker}: {content}"
    return content


def _extract_turns(sample_id: str, record: dict[str, Any]) -> Iterable[Turn]:
    conversation = record.get("conversation", record)

    if isinstance(conversation, list):
        yield from _turns_from_session_list(sample_id, "session_1", 1, conversation, None)
        return

    if not isinstance(conversation, dict):
        return

    session_keys: list[tuple[int, str, str | None]] = []
    for key in conversation:
        match = SESSION_RE.match(str(key))
        if match and isinstance(conversation.get(key), list):
            session_key = str(key)
            session_keys.append(
                (
                    int(match.group(1)),
                    session_key,
                    _string_or_none(conversation.get(f"{session_key}_date_time")),
                )
            )

    ordered_sessions = sorted(session_keys, key=_session_sort_key)
    for session_index, (_, session_key, timestamp) in enumerate(ordered_sessions, start=1):
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
        answer = item.get("answer", item.get("gold_answer", ""))
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


def _session_sort_key(item: tuple[int, str, str | None]) -> tuple[int, datetime | int]:
    session_number, _, timestamp = item
    if timestamp:
        for date_format in (
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %d %b, %Y",
            "%Y-%m-%d",
        ):
            try:
                return 0, datetime.strptime(timestamp, date_format)
            except ValueError:
                continue
    return 1, session_number
