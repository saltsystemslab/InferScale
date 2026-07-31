from __future__ import annotations

from typing import Any

import pytest

from locomo_jasper_bench.vector_types import SearchHit
from rag_bench.data_types import RagQuery
from rag_bench.prompting import (
    EMPTY_PASSAGES_TEXT,
    RAG_SYSTEM_PROMPT,
    build_rag_memory_token_ids,
    build_rag_query_messages,
    build_rag_query_tokens,
    calculate_rag_memory_budget,
    extract_rag_scaffold_token_ids,
    require_identical_token_ids,
    require_memory_within_budget,
    reverse_ranked_chunk_ids,
)
from rag_test_utils import CharTokenizer, TemplateTokenizer


class BosCharTokenizer(CharTokenizer):
    bos_token_id = 7

    def encode(self, text: str, add_special_tokens: bool = True, **_: Any) -> list[int]:
        token_ids = [ord(character) for character in text]
        return [self.bos_token_id, *token_ids] if add_special_tokens else token_ids


def _hit(chunk_id: str, rank: int) -> SearchHit:
    return SearchHit(
        id=chunk_id,
        payload={"chunk_id": chunk_id},
        score=float(-rank),
        distance=float(rank),
        rank=rank,
    )


def _query(question: str = "Who won?") -> RagQuery:
    return RagQuery(
        query_id="q0",
        question=question,
        gold_answer="x",
        question_type="inference_query",
        evidence=(),
    )


def test_scaffold_without_chat_template_uses_system_fallback() -> None:
    tokenizer = CharTokenizer()
    scaffold = extract_rag_scaffold_token_ids(tokenizer, block_size=16)

    header_text = tokenizer.decode(scaffold.header_token_ids)
    assert header_text.startswith("SYSTEM: ")
    assert RAG_SYSTEM_PROMPT in header_text
    assert scaffold.empty_memory_token_ids == tokenizer.encode(EMPTY_PASSAGES_TEXT)
    assert scaffold.memory_list_header_token_ids == []
    assert len(scaffold.footer_token_ids) >= 15


def test_scaffold_footer_pad_survives_content_trimming_template() -> None:
    scaffold = extract_rag_scaffold_token_ids(TemplateTokenizer(), block_size=16)

    assert len(scaffold.footer_token_ids) >= 15


def test_memory_token_ids_are_header_chunks_footer() -> None:
    scaffold = extract_rag_scaffold_token_ids(CharTokenizer(), block_size=4)
    chunk_a = [1, 2, 3]
    chunk_b = [4, 5]

    token_ids = build_rag_memory_token_ids(scaffold, [chunk_a, chunk_b])

    assert token_ids == (
        list(scaffold.header_token_ids) + chunk_a + chunk_b + list(scaffold.footer_token_ids)
    )


def test_empty_retrieval_renders_empty_passages_marker() -> None:
    scaffold = extract_rag_scaffold_token_ids(CharTokenizer(), block_size=4)

    token_ids = build_rag_memory_token_ids(scaffold, [])

    assert token_ids == (
        list(scaffold.header_token_ids)
        + list(scaffold.empty_memory_token_ids)
        + list(scaffold.footer_token_ids)
    )


def test_reverse_ranked_chunk_ids_dedupes_and_reverses() -> None:
    hits = [_hit("c1", 1), _hit("c2", 2), _hit("c1", 3), _hit("c3", 4)]

    assert reverse_ranked_chunk_ids(hits) == ["c3", "c2", "c1"]


def test_query_messages_pin_the_insufficient_phrase() -> None:
    messages = build_rag_query_messages(_query("Who won the case?"))

    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert "answer exactly: Insufficient information" in messages[0]["content"]
    assert "Question: Who won the case?" in messages[0]["content"]


def test_query_tokens_strip_duplicate_bos() -> None:
    tokenizer = BosCharTokenizer()
    memory_with_bos = [7, 100, 101]

    stripped = build_rag_query_tokens(tokenizer, memory_with_bos, _query())
    assert stripped.stripped_query_bos is True
    assert stripped.token_ids[0] != 7

    unstripped = build_rag_query_tokens(tokenizer, [100, 101], _query())
    assert unstripped.stripped_query_bos is False
    assert unstripped.token_ids[0] == 7


def test_memory_budget_math_and_overflow() -> None:
    assert (
        calculate_rag_memory_budget(
            query_token_count=10,
            max_position=100,
            max_model_len=1000,
            max_answer_tokens=20,
        )
        == 100
    )
    assert (
        calculate_rag_memory_budget(
            query_token_count=30,
            max_position=10000,
            max_model_len=100,
            max_answer_tokens=30,
        )
        == 40
    )
    with pytest.raises(RuntimeError, match="exceed kv_max_model_len"):
        calculate_rag_memory_budget(
            query_token_count=90,
            max_position=10000,
            max_model_len=100,
            max_answer_tokens=30,
        )


def test_over_budget_composition_raises_with_flags() -> None:
    require_memory_within_budget(10, 10, top_k=15, chunk_size=1024)
    with pytest.raises(RuntimeError, match="--top-k"):
        require_memory_within_budget(11, 10, top_k=15, chunk_size=1024)


def test_token_parity_guard_reports_first_mismatch() -> None:
    require_identical_token_ids([1, 2, 3], [1, 2, 3])
    with pytest.raises(RuntimeError, match="index=1"):
        require_identical_token_ids([1, 2, 3], [1, 9, 3])
    with pytest.raises(RuntimeError, match="index=2"):
        require_identical_token_ids([1, 2], [1, 2, 3])
