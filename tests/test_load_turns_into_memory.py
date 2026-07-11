from __future__ import annotations

from typing import Any

from locomo_jasper_bench.data import ConversationSample, Turn
from locomo_jasper_bench.retrieval.memory_builder import load_turns_into_memory


class _RecordingMemory:
    def __init__(self) -> None:
        self.add_calls: list[tuple[Any, dict[str, Any]]] = []

    def add(self, messages: Any, **kwargs: Any) -> None:
        self.add_calls.append((messages, kwargs))


def _sample() -> ConversationSample:
    return ConversationSample(
        sample_id="sample-1",
        turns=[
            Turn(
                sample_id="sample-1",
                session_id="session_1",
                session_index=1,
                turn_index=0,
                speaker="Alice",
                text="I like tea.",
            ),
            Turn(
                sample_id="sample-1",
                session_id="session_2",
                session_index=2,
                turn_index=0,
                speaker="Bob",
                text="I like coffee.",
            ),
        ],
        qa=[],
        raw={},
    )


def test_load_turns_into_memory_replays_raw_turns_without_inference() -> None:
    memory = _RecordingMemory()
    sample = _sample()

    added = load_turns_into_memory(memory, sample)

    assert added == 2
    assert len(memory.add_calls) == 2
    for (messages, kwargs), turn in zip(memory.add_calls, sample.turns):
        assert messages == [{"role": "user", "content": f"{turn.speaker}: {turn.text}"}]
        assert kwargs["infer"] is False
        assert kwargs["user_id"] == "sample-1"
        assert kwargs["metadata"]["turn_id"] == turn.id
        assert kwargs["metadata"]["session_id"] == turn.session_id
