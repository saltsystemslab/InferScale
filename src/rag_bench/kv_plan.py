from __future__ import annotations

from collections.abc import Sequence

from locomo_jasper_bench.kv.types import FactContextEncodingPlan

from .data_types import RagChunk


def build_chunk_context_encoding_plan(
    target: RagChunk,
    doc_chunks: Sequence[RagChunk],
    *,
    context_window: int,
    max_input_tokens: int,
) -> FactContextEncodingPlan:
    """Encoding plan for one chunk with its preceding-chunk context prefix.

    The context is up to context_window chunks of the SAME document that
    immediately precede the target (document order), used as an encode-only
    prefix: the forward pass sees context + target and only the target span's
    KV is sliced out and stored. Overflow truncates the oldest context tokens
    first and never the target, mirroring kv/context.py's
    build_fact_context_encoding_plan.
    """
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be >= 1.")

    target_token_ids = list(target.token_ids)
    if not target_token_ids:
        raise RuntimeError(f"Corpus chunk tokenized to zero tokens: {target.chunk_id}")
    if len(target_token_ids) > max_input_tokens:
        raise RuntimeError(
            f"Corpus chunk {target.chunk_id} has {len(target_token_ids)} tokens, "
            f"exceeding kv_max_position={max_input_tokens} without context."
        )

    position: int | None = None
    for index, chunk in enumerate(doc_chunks):
        if chunk.doc_id != target.doc_id:
            raise RuntimeError(
                f"doc_chunks for {target.chunk_id} contains chunk {chunk.chunk_id} "
                "from a different document."
            )
        if chunk.chunk_id == target.chunk_id:
            position = index
    if position is None:
        raise RuntimeError(
            f"Target chunk {target.chunk_id} is not present in its document's chunk list."
        )

    context_chunks = (
        list(doc_chunks[max(0, position - context_window) : position])
        if context_window
        else []
    )
    context_token_ids: list[int] = []
    for context_chunk in context_chunks:
        context_token_ids.extend(context_chunk.token_ids)

    raw_context_tokens = len(context_token_ids)
    overflow = raw_context_tokens + len(target_token_ids) - max_input_tokens
    context_truncated_tokens = max(0, overflow)
    if context_truncated_tokens:
        context_token_ids = context_token_ids[context_truncated_tokens:]

    input_token_ids = context_token_ids + target_token_ids
    slice_start = len(context_token_ids)
    return FactContextEncodingPlan(
        memory_id=target.chunk_id,
        target_token_ids=target_token_ids,
        context_token_ids=context_token_ids,
        input_token_ids=input_token_ids,
        slice_start=slice_start,
        slice_end=len(input_token_ids),
        context_turn_ids=tuple(chunk.chunk_id for chunk in context_chunks),
        raw_context_tokens=raw_context_tokens,
        context_truncated_tokens=context_truncated_tokens,
    )
