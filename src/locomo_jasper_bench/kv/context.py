from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..data import ConversationSample, Turn, format_turn_for_memory
from .tokenization import encode_text_no_special
from .types import ContextEncodingPlan


def build_turn_context_encoding_plan(
    tokenizer: Any,
    sample: ConversationSample,
    turn: Turn,
    *,
    context_window: int,
    max_input_tokens: int,
    turn_token_ids: Mapping[str, list[int]] | None = None,
) -> ContextEncodingPlan:
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be >= 1.")

    target_text = format_memory_turn(turn)
    target_token_ids = _turn_token_ids(tokenizer, turn, target_text, turn_token_ids)
    if not target_token_ids:
        raise RuntimeError(f"Memory chunk tokenized to zero tokens: {turn.id}")
    if len(target_token_ids) > max_input_tokens:
        raise RuntimeError(
            f"Memory turn {turn.id} has {len(target_token_ids)} tokens, "
            f"exceeding kv_max_position={max_input_tokens} even without context."
        )

    context_token_ids: list[int] = []
    for context_turn in previous_session_context_turns(sample, turn, context_window):
        context_token_ids.extend(
            _turn_token_ids(
                tokenizer,
                context_turn,
                format_memory_turn(context_turn),
                turn_token_ids,
            )
        )

    raw_context_prefix_tokens = len(context_token_ids)
    overflow = len(context_token_ids) + len(target_token_ids) - max_input_tokens
    context_prefix_truncated_tokens = max(0, overflow)
    if context_prefix_truncated_tokens:
        context_token_ids = context_token_ids[context_prefix_truncated_tokens:]

    input_token_ids = context_token_ids + target_token_ids
    slice_start = len(context_token_ids)
    slice_end = len(input_token_ids)
    return ContextEncodingPlan(
        turn_id=turn.id,
        target_text=target_text,
        target_token_ids=target_token_ids,
        context_token_ids=context_token_ids,
        input_token_ids=input_token_ids,
        slice_start=slice_start,
        slice_end=slice_end,
        raw_context_prefix_tokens=raw_context_prefix_tokens,
        context_prefix_truncated_tokens=context_prefix_truncated_tokens,
    )


def previous_session_context_turns(sample: ConversationSample, turn: Turn, context_window: int) -> list[Turn]:
    if context_window <= 0:
        return []
    first_session_index = turn.session_index - context_window
    return [
        candidate
        for candidate in sample.turns
        if first_session_index <= candidate.session_index < turn.session_index
    ]


def format_memory_turn(turn: Turn) -> str:
    return format_turn_for_memory(turn).strip() + "\n"


def _turn_token_ids(
    tokenizer: Any,
    turn: Turn,
    text: str,
    turn_token_ids: Mapping[str, list[int]] | None,
) -> list[int]:
    if turn_token_ids is not None and turn.id in turn_token_ids:
        return list(turn_token_ids[turn.id])
    return encode_text_no_special(tokenizer, text)
