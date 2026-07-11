from __future__ import annotations

import pytest

from locomo_jasper_bench.data import ConversationSample, Turn
from locomo_jasper_bench.kv.context import (
    build_turn_context_encoding_plan,
    previous_session_context_turns,
)


def _turn(session_index: int, turn_index: int) -> Turn:
    return Turn(
        sample_id="sample-1",
        session_id=f"session_{session_index}",
        session_index=session_index,
        turn_index=turn_index,
        speaker="speaker",
        text=f"session {session_index}, turn {turn_index}",
    )


def _sample() -> ConversationSample:
    return ConversationSample(
        sample_id="sample-1",
        turns=[
            _turn(1, 0),
            _turn(1, 1),
            _turn(2, 0),
            _turn(2, 1),
            _turn(3, 0),
        ],
        qa=[],
        raw={},
    )


def test_context_window_selects_all_turns_of_previous_sessions() -> None:
    sample = _sample()

    selected = previous_session_context_turns(sample, sample.turns[4], context_window=1)

    assert [turn.id for turn in selected] == [sample.turns[2].id, sample.turns[3].id]


def test_context_window_spans_multiple_previous_sessions() -> None:
    sample = _sample()

    selected = previous_session_context_turns(sample, sample.turns[4], context_window=2)

    assert [turn.id for turn in selected] == [turn.id for turn in sample.turns[:4]]


def test_context_window_excludes_same_session_turns() -> None:
    sample = _sample()

    selected = previous_session_context_turns(sample, sample.turns[3], context_window=1)

    assert [turn.id for turn in selected] == [sample.turns[0].id, sample.turns[1].id]
    assert sample.turns[2].id not in [turn.id for turn in selected]


def test_context_window_uses_available_sessions_at_the_start() -> None:
    sample = _sample()

    selected = previous_session_context_turns(sample, sample.turns[1], context_window=5)

    assert selected == []


def test_zero_context_window_returns_no_context() -> None:
    sample = _sample()

    assert previous_session_context_turns(sample, sample.turns[3], context_window=0) == []


def test_context_window_rejects_negative_values() -> None:
    sample = _sample()

    with pytest.raises(ValueError, match="context_window must be >= 0"):
        previous_session_context_turns(sample, sample.turns[0], context_window=-1)


def test_encoding_plan_uses_cached_tokens_in_session_order_and_truncates_from_left() -> None:
    sample = _sample()
    target = sample.turns[3]
    cached_tokens = {
        sample.turns[0].id: [11, 12],
        sample.turns[1].id: [20],
        target.id: [30, 31],
    }

    plan = build_turn_context_encoding_plan(
        object(),
        sample,
        target,
        context_window=1,
        max_input_tokens=4,
        turn_token_ids=cached_tokens,
    )

    assert plan.target_token_ids == [30, 31]
    assert plan.raw_context_prefix_tokens == 3
    assert plan.context_prefix_truncated_tokens == 1
    assert plan.context_token_ids == [12, 20]
    assert plan.input_token_ids == [12, 20, 30, 31]
    assert (plan.slice_start, plan.slice_end) == (2, 4)
