"""The encoding plan of one corpus chunk: its preceding chunks of the same document as prefix."""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.rag.data_types import RagChunk
from inferscale.v1.kv.plan import build_encoding_plan
from inferscale.v1.types import EncodingPlan


def build_chunk_context_encoding_plan(
    target: RagChunk,
    doc_chunks: Sequence[RagChunk],
    *,
    context_window: int,
    max_input_tokens: int,
) -> EncodingPlan:
    """Encoding plan for one chunk with its preceding-chunk context prefix.

    The context is up to context_window chunks of the same document that
    immediately precede the target (document order), used as an encode-only
    prefix: the forward pass sees context + target and only the target span's
    KV is sliced out and stored. Overflow truncates the oldest context tokens
    first and never the target, as the memory benchmark's turn-window plan in
    benchmarks/memory/context.py does.
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
    position = _position_in_document(target, doc_chunks)
    context_chunks = list(doc_chunks[max(0, position - context_window) : position])
    return build_encoding_plan(
        target.chunk_id,
        target_token_ids,
        context_token_ids=[token for chunk in context_chunks for token in chunk.token_ids],
        context_ids=tuple(chunk.chunk_id for chunk in context_chunks),
        max_input_tokens=max_input_tokens,
    )


def _position_in_document(target: RagChunk, doc_chunks: Sequence[RagChunk]) -> int:
    for chunk in doc_chunks:
        if chunk.doc_id != target.doc_id:
            raise RuntimeError(
                f"doc_chunks for {target.chunk_id} contains chunk {chunk.chunk_id} "
                "from a different document."
            )
    try:
        return [chunk.chunk_id for chunk in doc_chunks].index(target.chunk_id)
    except ValueError:
        raise RuntimeError(
            f"Target chunk {target.chunk_id} is not present in its document's chunk list."
        ) from None
