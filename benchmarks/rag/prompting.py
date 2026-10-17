"""The RAG prompt: the dataset's scaffold around the passages, and the templated question."""

from __future__ import annotations

from typing import Any

from inferscale.v1.kv.prompt import (
    QueryTokens,
    ScaffoldTokens,
    build_query_tokens,
    build_scaffold_tokens,
)

from .data_types import RagQuery

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


def build_rag_query_messages(
    query: RagQuery,
    *,
    answer_instruction: str,
) -> list[dict[str, str]]:
    if not answer_instruction:
        raise ValueError("answer_instruction must be non-empty.")
    return [{"role": "user", "content": f"{answer_instruction}Question: {query.question}"}]


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


def require_memory_within_budget(
    memory_token_count: int,
    memory_token_budget: int,
    *,
    top_k: int,
    chunk_size: int,
) -> None:
    """Fail on an over-budget composition, naming the settings to lower.

    Retrieved chunks are never dropped silently.
    """
    if memory_token_count <= memory_token_budget:
        return
    raise RuntimeError(
        f"Composed passages need {memory_token_count} tokens but the memory budget is "
        f"{memory_token_budget} (top_k={top_k}, chunk_size={chunk_size}). Lower top_k "
        "or chunk_size, or raise kv max_position and max_model_len."
    )
