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
    newest_context = context[context_truncated_tokens:]

    input_token_ids = newest_context + target
    return EncodingPlan(
        chunk_id=chunk_id,
        target_token_ids=target,
        context_token_ids=newest_context,
        input_token_ids=input_token_ids,
        slice_start=len(newest_context),
        slice_end=len(input_token_ids),
        context_ids=tuple(context_ids),
        raw_context_tokens=raw_context_tokens,
        context_truncated_tokens=context_truncated_tokens,
    )
