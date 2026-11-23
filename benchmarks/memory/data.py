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
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
ASSISTANT_SPEAKERS = frozenset({"assistant", "ai", "bot"})
DATED_FIRST = 0
UNDATED_LAST = 1


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
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"A LoCoMo file must be a JSON list of samples: {path}")
    selected = records if max_samples is None else records[:max_samples]
    return [_sample(record) for record in selected]


def format_turn_for_memory(turn: Turn) -> str:
    utterance = turn.text.strip()
    turn_record = turn.raw if isinstance(turn.raw, dict) else {}
    image_query = str(turn_record.get("query") or "").strip()
    image_caption = str(turn.image_caption or "").strip()
    if image_query and image_caption:
        photo_tag = f"[Sharing image - query: {image_query}. The image shows: {image_caption}]"
    elif image_query:
        photo_tag = f"[Sharing image - query for: {image_query}]"
    elif image_caption:
        photo_tag = f"[Sharing image that shows: {image_caption}]"
    else:
        photo_tag = ""
    content = " ".join(part for part in (utterance, photo_tag) if part)
    if turn.speaker:
        return f"{turn.speaker}: {content}"
    return content


def locomo_turn_role(sample: ConversationSample, turn: Turn) -> str:
    conversation = _conversation_record(sample)
    speaker_a = str(conversation.get("speaker_a") or "").strip()
    speaker_b = str(conversation.get("speaker_b") or "").strip()
    if speaker_a and turn.speaker == speaker_a:
        return "user"
    if speaker_b and turn.speaker == speaker_b:
        return "assistant"
    if turn.speaker.strip().lower() in ASSISTANT_SPEAKERS:
        return "assistant"
    return "user"


def locomo_timestamp(value: str | None) -> tuple[int, str]:
    text = str(value or "").strip()
    parsed = _parse_timestamp(text) if text else UNIX_EPOCH
    moment = _in_utc(parsed)
    return int(moment.timestamp()), moment.isoformat()


def sample_fingerprint(sample: ConversationSample) -> str:
    conversation = _conversation_record(sample)
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


def _conversation_record(sample: ConversationSample) -> dict[str, Any]:
    conversation = sample.raw.get("conversation", sample.raw)
    return conversation if isinstance(conversation, dict) else {}


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
    qa = [
        _question(sample_id, index, question_record)
        for index, question_record in enumerate(record.get("qa", []))
    ]
    return ConversationSample(sample_id=sample_id, turns=turns, qa=qa, raw=record)


def _sessions_in_date_order(conversation: dict[str, Any]) -> list[tuple[str, str | None]]:
    sessions = []
    for key in conversation:
        session_match = SESSION_KEY.match(key)
        if session_match is None:
            continue
        date_text = conversation.get(f"{key}_date_time")
        session_date = _parse_locomo_date(date_text) if date_text else None
        if session_date is not None:
            position = (DATED_FIRST, session_date)
        else:
            position = (UNDATED_LAST, int(session_match.group(1)))
        sessions.append((position, key, date_text))
    sessions.sort(key=lambda session: session[0])
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
            speaker=str(turn_record.get("speaker") or ""),
            text=str(turn_record.get("text") or "").strip(),
            timestamp=date_text,
            image_caption=_text_or_none(turn_record.get("blip_caption")),
            raw=turn_record,
        )
        for turn_index, turn_record in enumerate(conversation[session_key])
    ]


def _question(sample_id: str, index: int, question_record: dict[str, Any]) -> QuestionAnswer:
    return QuestionAnswer(
        sample_id=sample_id,
        question_id=f"q{index}",
        question=str(question_record["question"]).strip(),
        answer=str(question_record.get("answer", "")),
        category=str(question_record["category"]),
        evidence=question_record.get("evidence"),
        raw=question_record,
    )


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_timestamp(text: str) -> datetime:
    moment = _parse_iso(text) or _parse_locomo_date(text)
    if moment is None:
        raise ValueError(f"Unsupported LoCoMo timestamp: {text!r}.")
    return moment


def _in_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


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
