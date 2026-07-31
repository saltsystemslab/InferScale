from __future__ import annotations

import pytest

from rag_bench.data_types import RagChunk
from rag_bench.kv_plan import build_chunk_context_encoding_plan


def _chunk(doc_id: str, index: int, token_ids: list[int]) -> RagChunk:
    return RagChunk(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        chunk_index=index,
        token_ids=token_ids,
        text="",
    )


def _doc_chunks() -> list[RagChunk]:
    return [
        _chunk("dA", 0, [10, 11, 12, 13]),
        _chunk("dA", 1, [20, 21, 22, 23]),
        _chunk("dA", 2, [30, 31, 32, 33]),
        _chunk("dA", 3, [40, 41, 42]),
    ]


def test_window_selects_preceding_chunks_of_same_document() -> None:
    chunks = _doc_chunks()

    plan = build_chunk_context_encoding_plan(
        chunks[3], chunks, context_window=2, max_input_tokens=10_000
    )

    assert plan.memory_id == "dA:3"
    assert plan.context_token_ids == [20, 21, 22, 23, 30, 31, 32, 33]
    assert plan.target_token_ids == [40, 41, 42]
    assert plan.input_token_ids == plan.context_token_ids + plan.target_token_ids
    assert (plan.slice_start, plan.slice_end) == (8, 11)
    assert plan.context_turn_ids == ("dA:1", "dA:2")
    assert plan.context_truncated_tokens == 0


def test_document_start_gets_shorter_or_empty_context() -> None:
    chunks = _doc_chunks()

    first = build_chunk_context_encoding_plan(
        chunks[0], chunks, context_window=5, max_input_tokens=10_000
    )
    second = build_chunk_context_encoding_plan(
        chunks[1], chunks, context_window=5, max_input_tokens=10_000
    )

    assert first.context_token_ids == [] and first.context_turn_ids == ()
    assert first.slice_start == 0
    assert second.context_token_ids == [10, 11, 12, 13]
    assert second.context_turn_ids == ("dA:0",)


def test_zero_window_has_no_context() -> None:
    chunks = _doc_chunks()

    plan = build_chunk_context_encoding_plan(
        chunks[2], chunks, context_window=0, max_input_tokens=10_000
    )

    assert plan.context_token_ids == []
    assert plan.input_token_ids == [30, 31, 32, 33]


def test_overflow_truncates_oldest_context_tokens_never_the_target() -> None:
    chunks = _doc_chunks()
    # Context is 8 tokens, target is 3; cap forces a 2-token overflow.
    plan = build_chunk_context_encoding_plan(
        chunks[3], chunks, context_window=2, max_input_tokens=9
    )

    assert plan.raw_context_tokens == 8
    assert plan.context_truncated_tokens == 2
    assert plan.context_token_ids == [22, 23, 30, 31, 32, 33]
    assert plan.target_token_ids == [40, 41, 42]
    assert plan.slice_start == 6
    assert plan.context_turn_ids == ("dA:1", "dA:2")


def test_target_larger_than_limit_raises() -> None:
    chunks = _doc_chunks()

    with pytest.raises(RuntimeError, match="without context"):
        build_chunk_context_encoding_plan(
            chunks[0], chunks, context_window=2, max_input_tokens=3
        )


def test_foreign_document_chunk_raises() -> None:
    chunks = _doc_chunks()
    chunks[1] = _chunk("dB", 1, [99])

    with pytest.raises(RuntimeError, match="different document"):
        build_chunk_context_encoding_plan(
            chunks[3], chunks, context_window=2, max_input_tokens=100
        )


def test_target_missing_from_doc_chunks_raises() -> None:
    chunks = _doc_chunks()
    stray = _chunk("dA", 9, [1])

    with pytest.raises(RuntimeError, match="not present"):
        build_chunk_context_encoding_plan(stray, chunks, context_window=2, max_input_tokens=100)


def test_invalid_arguments_raise() -> None:
    chunks = _doc_chunks()

    with pytest.raises(ValueError, match="context_window"):
        build_chunk_context_encoding_plan(
            chunks[0], chunks, context_window=-1, max_input_tokens=100
        )
    with pytest.raises(ValueError, match="max_input_tokens"):
        build_chunk_context_encoding_plan(
            chunks[0], chunks, context_window=0, max_input_tokens=0
        )
