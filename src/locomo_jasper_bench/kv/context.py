from __future__ import annotations

from typing import Any

from ..data import ConversationSample, SessionChunk, format_session_for_memory
from .tokenization import encode_text_no_special
from .types import ContextEncodingPlan


def build_session_context_encoding_plan(
    tokenizer: Any,
    sample: ConversationSample,
    session: SessionChunk,
    *,
    context_window: int,
    max_input_tokens: int,
) -> ContextEncodingPlan:
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be >= 1.")

    target_text = format_memory_session(session)
    target_token_ids = encode_text_no_special(tokenizer, target_text)
    if not target_token_ids:
        raise RuntimeError(f"Memory chunk tokenized to zero tokens: {session.id}")
    if len(target_token_ids) > max_input_tokens:
        raise RuntimeError(
            f"Memory session {session.id} has {len(target_token_ids)} tokens, "
            f"exceeding kv_max_position={max_input_tokens} even without context."
        )

    context_token_ids: list[int] = []
    for context_session in previous_session_context_chunks(sample, session, context_window):
        context_token_ids.extend(encode_text_no_special(tokenizer, format_memory_session(context_session)))

    raw_context_prefix_tokens = len(context_token_ids)
    overflow = len(context_token_ids) + len(target_token_ids) - max_input_tokens
    context_prefix_truncated_tokens = max(0, overflow)
    if context_prefix_truncated_tokens:
        context_token_ids = context_token_ids[context_prefix_truncated_tokens:]

    input_token_ids = context_token_ids + target_token_ids
    slice_start = len(context_token_ids)
    slice_end = len(input_token_ids)
    return ContextEncodingPlan(
        chunk_id=session.id,
        target_text=target_text,
        target_token_ids=target_token_ids,
        context_token_ids=context_token_ids,
        input_token_ids=input_token_ids,
        slice_start=slice_start,
        slice_end=slice_end,
        raw_context_prefix_tokens=raw_context_prefix_tokens,
        context_prefix_truncated_tokens=context_prefix_truncated_tokens,
    )


def previous_session_context_chunks(
    sample: ConversationSample,
    session: SessionChunk,
    context_window: int,
) -> list[SessionChunk]:
    if context_window <= 0:
        return []
    first_session_index = session.session_index - context_window
    return [
        candidate
        for candidate in sample.sessions
        if first_session_index <= candidate.session_index < session.session_index
    ]


def format_memory_session(session: SessionChunk) -> str:
    return format_session_for_memory(session).strip() + "\n\n"
