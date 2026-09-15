from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from inferscale.v1.kv.compose import reverse_ranked_ids
from inferscale.v1.kv.prompt import (
    QueryTokens,
    ScaffoldTokens,
    build_memory_token_ids,
    build_query_tokens,
    build_scaffold_tokens,
    memory_token_budget,
)
from inferscale.v1.types import SearchHit

from .data_types import RagPromptProfile, RagQuery

RAG_TEMPLATE_PLACEHOLDER = "<<<RAG_JASPER_PASSAGES_GO_HERE>>>"
EMPTY_PASSAGES_TEXT = "(No relevant passages found)\n"


def extract_rag_scaffold_token_ids(
    tokenizer: Any,
    *,
    system_prompt: str,
    block_size: int = 16,
) -> ScaffoldTokens:
    """Chat-templated scaffold split around the passages placeholder.

    The system prompt is dataset-specific (RagPromptProfile.system_prompt).
    """
    return build_scaffold_tokens(
        tokenizer,
        system_prompt=system_prompt,
        empty_text=EMPTY_PASSAGES_TEXT,
        block_size=block_size,
        placeholder=RAG_TEMPLATE_PLACEHOLDER,
    )


def reverse_ranked_chunk_ids(hits: Sequence[SearchHit]) -> list[str]:
    """Unique retrieved chunk ids with the best-ranked chunk last."""
    return reverse_ranked_ids(hits)


def build_rag_memory_token_ids(
    scaffold: ScaffoldTokens,
    ordered_chunk_token_ids: Sequence[Sequence[int]],
) -> list[int]:
    return build_memory_token_ids(scaffold, ordered_chunk_token_ids)


def build_rag_query_messages(
    query: RagQuery,
    *,
    answer_instruction: str,
) -> list[dict[str, str]]:
    if not answer_instruction:
        raise ValueError("answer_instruction must be non-empty.")
    return [
        {
            "role": "user",
            "content": f"{answer_instruction}Question: {query.question}",
        }
    ]


def build_rag_query_tokens(
    tokenizer: Any,
    memory_token_ids: list[int],
    query: RagQuery,
    *,
    answer_instruction: str,
) -> QueryTokens:
    return build_query_tokens(
        tokenizer,
        memory_token_ids,
        build_rag_query_messages(query, answer_instruction=answer_instruction),
    )


def calculate_rag_memory_budget(
    *,
    query_token_count: int,
    max_position: int,
    max_model_len: int,
    max_answer_tokens: int,
) -> int:
    return memory_token_budget(
        query_token_count=query_token_count,
        max_position=max_position,
        max_model_len=max_model_len,
        max_answer_tokens=max_answer_tokens,
    )


def require_memory_within_budget(
    memory_token_count: int,
    memory_token_budget: int,
    *,
    top_k: int,
    chunk_size: int,
) -> None:
    """Strict fail on over-budget composition; retrieved chunks are never dropped."""
    if memory_token_count <= memory_token_budget:
        return
    raise RuntimeError(
        f"Composed passages need {memory_token_count} tokens but the memory budget is "
        f"{memory_token_budget} (top_k={top_k}, chunk_size={chunk_size}). Lower top_k "
        "or chunk_size, or raise kv max_position and max_model_len."
    )
