from __future__ import annotations

from typing import Any

import pytest

from benchmarks.memory.data import (
    ConversationSample,
    Turn,
    format_turn_for_memory,
    sample_fingerprint,
)


def turn(**fields: Any) -> Turn:
    return Turn(
        **{
            "sample_id": "conv-1",
            "session_id": "session_1",
            "session_index": 1,
            "turn_index": 0,
            "speaker": "Alice",
            "text": "Look at this.",
            **fields,
        }
    )


def sample_with(raw: dict[str, Any], turns: list[Turn] | None = None) -> ConversationSample:
    return ConversationSample("conv-1", turns or [], [], raw)


@pytest.mark.parametrize(
    "fields,memory_text",
    [
        ({}, "Alice: Look at this."),
        ({"raw": {"query": "kayak"}}, "Alice: Look at this. [Sharing image - query for: kayak]"),
        (
            {"image_caption": "a green kayak"},
            "Alice: Look at this. [Sharing image that shows: a green kayak]",
        ),
    ],
)
def test_memory_text_is_the_speaker_the_utterance_and_the_shared_image(fields, memory_text):
    assert format_turn_for_memory(turn(**fields)) == memory_text


def fingerprinted(raw: dict[str, Any] | None = None, **turn_fields: Any) -> str:
    turns = [
        turn(**{"timestamp": "1:56 pm on 8 May, 2023", "image_caption": "a kayak", **turn_fields}),
        turn(turn_index=1, speaker="Bob", text="Hi."),
    ]
    speakers = {"conversation": {"speaker_a": "Alice", "speaker_b": "Bob"}}
    return sample_fingerprint(sample_with(speakers if raw is None else raw, turns))


def test_fingerprint_is_a_stable_digest_of_the_speakers_and_turns():
    assert fingerprinted() == "a9e0f0d8936c83558c44"
    assert fingerprinted({"speaker_a": "Alice", "speaker_b": "Bob"}) == fingerprinted()
    assert fingerprinted({}) == "225a64858b242514a1ea"
    assert fingerprinted({"conversation": "not an object"}) == fingerprinted({})
    assert sample_fingerprint(sample_with({})) == "d23054ba299861fd5728"
