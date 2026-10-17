"""LoCoMo dataset records and the dataset-specific helpers the benchmark shares.

A LoCoMo file is a JSON list of samples. Each sample has a ``sample_id``, a
``conversation`` object, and a ``qa`` list. The conversation holds
``speaker_a`` and ``speaker_b`` and, per session N, a ``session_N`` list of
turns plus a ``session_N_date_time`` string such as "1:56 pm on 8 May, 2023".
A turn has ``speaker``, ``dia_id``, and ``text``; image turns add
``img_url``, ``blip_caption``, and ``query``. A question has ``question``,
``evidence`` (dialogue ids), an integer ``category`` from 1 to 5, and an
``answer``; adversarial questions (category 5) carry ``adversarial_answer``
instead. Questions have no id in the file, so they are numbered ``q0``,
``q1``, ... in file order.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSION_KEY = re.compile(r"^session_(\d+)$")
LOCOMO_DATE_FORMATS = ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y")


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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"A LoCoMo file must be a JSON list of samples: {path}")
    records = data if max_samples is None else data[:max_samples]
    return [_sample(record) for record in records]


def format_turn_for_memory(turn: Turn) -> str:
    """The turn text Mem0 ingests: speaker, utterance, and the shared image, if any."""
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


def locomo_turn_role(sample: ConversationSample, turn: Turn) -> str:
    """Mem0's chat role for a turn: speaker_a is the user and speaker_b the assistant."""
    conversation = sample.raw.get("conversation", sample.raw)
    if not isinstance(conversation, dict):
        conversation = {}
    speaker_a = str(conversation.get("speaker_a") or "").strip()
    speaker_b = str(conversation.get("speaker_b") or "").strip()
    if speaker_a and turn.speaker == speaker_a:
        return "user"
    if speaker_b and turn.speaker == speaker_b:
        return "assistant"
    normalized = turn.speaker.strip().lower()
    if normalized in {"assistant", "ai", "bot"}:
        return "assistant"
    return "user"


def locomo_timestamp(value: str | None) -> tuple[int, str]:
    """A UTC epoch and ISO pair for a session date; records without a date use the Unix epoch."""
    text = str(value or "").strip()
    if not text:
        parsed = datetime(1970, 1, 1, tzinfo=timezone.utc)
    else:
        parsed = _parse_iso(text) or _parse_locomo_date(text)
        if parsed is None:
            raise ValueError(f"Unsupported LoCoMo timestamp: {text!r}.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp()), parsed.isoformat()


def sample_fingerprint(sample: ConversationSample) -> str:
    """A short digest of the conversation content, part of every per-sample cache key."""
    conversation = sample.raw.get("conversation", sample.raw)
    if not isinstance(conversation, dict):
        conversation = {}
    canonical = json.dumps(
        {
            "sample_id": sample.sample_id,
            "speaker_a": conversation.get("speaker_a"),
            "speaker_b": conversation.get("speaker_b"),
            "turns": [
                {
                    "id": turn.id,
                    "session_index": turn.session_index,
                    "turn_index": turn.turn_index,
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "timestamp": turn.timestamp,
                    "image_caption": turn.image_caption,
                }
                for turn in sample.turns
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _sample(record: dict[str, Any]) -> ConversationSample:
    sample_id = str(record["sample_id"])
    conversation = record["conversation"]
    turns = [
        turn
        for session_index, (session_key, date_text) in enumerate(
            _sessions_in_date_order(conversation), start=1
        )
        for turn in _session_turns(sample_id, session_key, session_index, conversation, date_text)
    ]
    qa = [_question(sample_id, index, item) for index, item in enumerate(record.get("qa", []))]
    return ConversationSample(sample_id=sample_id, turns=turns, qa=qa, raw=record)


def _sessions_in_date_order(conversation: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Session keys and dates, earliest first; undated sessions keep their numeric order."""
    sessions = []
    for key in conversation:
        match = SESSION_KEY.match(key)
        if match is None:
            continue
        date_text = conversation.get(f"{key}_date_time")
        parsed = _parse_locomo_date(date_text) if date_text else None
        order = (0, parsed) if parsed is not None else (1, int(match.group(1)))
        sessions.append((order, key, date_text))
    sessions.sort(key=lambda item: item[0])
    return [(key, date_text) for _, key, date_text in sessions]


def _session_turns(
    sample_id: str,
    session_key: str,
    session_index: int,
    conversation: dict[str, Any],
    date_text: str | None,
) -> list[Turn]:
    return [
        Turn(
            sample_id=sample_id,
            session_id=session_key,
            session_index=session_index,
            turn_index=turn_index,
            speaker=str(turn.get("speaker") or ""),
            text=str(turn.get("text") or "").strip(),
            timestamp=date_text,
            image_caption=_text_or_none(turn.get("blip_caption")),
            raw=turn,
        )
        for turn_index, turn in enumerate(conversation[session_key])
    ]


def _question(sample_id: str, index: int, item: dict[str, Any]) -> QuestionAnswer:
    return QuestionAnswer(
        sample_id=sample_id,
        question_id=f"q{index}",
        question=str(item["question"]).strip(),
        answer=str(item.get("answer", "")),
        category=str(item["category"]),
        evidence=item.get("evidence"),
        raw=item,
    )


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_iso(text: str) -> datetime | None:
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso_text)
    except ValueError:
        return None


def _parse_locomo_date(text: str) -> datetime | None:
    for date_format in LOCOMO_DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None
