"""Encoding plans: an optional encoding-only context prefix plus the target chunk."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import EncodingPlan


def build_encoding_plan(
    chunk_id: str,
    target_token_ids: Sequence[int],
    *,
    context_token_ids: Sequence[int] = (),
    context_ids: Sequence[str] = (),
    max_input_tokens: int,
) -> EncodingPlan:
    """Plan one forward pass whose KV is kept only for the target span.

    The context prefix conditions the target's KV and is sliced away after
    the forward pass. When context plus target exceed ``max_input_tokens``
    the oldest context tokens are dropped first; the target is never cut.
    """
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be >= 1.")

    target = list(target_token_ids)
    if not target:
        raise RuntimeError(f"Chunk {chunk_id} tokenized to zero tokens.")
    if len(target) > max_input_tokens:
        raise RuntimeError(
            f"Chunk {chunk_id} has {len(target)} tokens, "
            f"exceeding max_position={max_input_tokens} without context."
        )

    context = list(context_token_ids)
    raw_context_tokens = len(context)
    overflow = raw_context_tokens + len(target) - max_input_tokens
    context_truncated_tokens = max(0, overflow)
    if context_truncated_tokens:
        context = context[context_truncated_tokens:]

    input_token_ids = context + target
    return EncodingPlan(
        chunk_id=chunk_id,
        target_token_ids=target,
        context_token_ids=context,
        input_token_ids=input_token_ids,
        slice_start=len(context),
        slice_end=len(input_token_ids),
        context_ids=tuple(context_ids),
        raw_context_tokens=raw_context_tokens,
        context_truncated_tokens=context_truncated_tokens,
    )
