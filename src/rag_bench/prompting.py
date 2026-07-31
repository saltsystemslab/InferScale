from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from locomo_jasper_bench.kv.prompting import (
    KvQueryTokens,
    MemoryScaffoldTokens,
    apply_chat_template_non_thinking,
    strip_duplicate_query_bos,
    tokenize_messages,
)
from locomo_jasper_bench.kv.tokenization import encode_text_no_special
from locomo_jasper_bench.vector_types import SearchHit

from .data_types import RagPromptProfile, RagQuery

RAG_TEMPLATE_PLACEHOLDER = "<<<RAG_JASPER_PASSAGES_GO_HERE>>>"
EMPTY_PASSAGES_TEXT = "(No relevant passages found)\n"


def extract_rag_scaffold_token_ids(
    tokenizer: Any,
    *,
    system_prompt: str,
    block_size: int = 16,
) -> MemoryScaffoldTokens:
    """Chat-templated scaffold split around the passages placeholder.

    The system prompt is dataset-specific (RagPromptProfile.system_prompt).
    Same invariants as the LoCoMo scaffold (kv/prompting.py): the footer is
    padded at the token level with newline tokens to at least block_size - 1
    tokens, so injected chunk tokens never land in the recomputed KV tail.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if not system_prompt:
        raise ValueError("system_prompt must be non-empty.")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        templated = apply_chat_template_non_thinking(
            tokenizer,
            [{"role": "system", "content": system_prompt + RAG_TEMPLATE_PLACEHOLDER}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        templated = f"SYSTEM: {system_prompt}{RAG_TEMPLATE_PLACEHOLDER}"

    if RAG_TEMPLATE_PLACEHOLDER not in templated:
        raise RuntimeError("Answer prompt chat template removed the passages placeholder.")
    header_text, footer_text = templated.split(RAG_TEMPLATE_PLACEHOLDER, 1)
    header_token_ids = encode_text_no_special(tokenizer, header_text)
    empty_passages_token_ids = encode_text_no_special(tokenizer, EMPTY_PASSAGES_TEXT)
    footer_close_token_ids = encode_text_no_special(tokenizer, footer_text)
    newline_token_ids = encode_text_no_special(tokenizer, "\n")
    if not newline_token_ids:
        raise RuntimeError("Tokenizer produced no tokens for a newline pad.")
    pad_repeats = max(0, block_size - 1 - len(footer_close_token_ids))
    footer_token_ids = newline_token_ids * pad_repeats + footer_close_token_ids
    if not header_token_ids or not empty_passages_token_ids:
        raise RuntimeError(
            f"Empty RAG scaffold tokens: header={len(header_token_ids)} "
            f"empty_passages={len(empty_passages_token_ids)} footer={len(footer_token_ids)}."
        )
    if len(footer_token_ids) < block_size - 1:
        raise RuntimeError(
            "The passages scaffold does not contain enough trailing whitespace to keep "
            f"retrieved chunks outside the recomputed KV tail: {len(footer_token_ids)} "
            f"footer token(s); need at least {block_size - 1}."
        )
    return MemoryScaffoldTokens(
        header_token_ids=header_token_ids,
        memory_list_header_token_ids=[],
        empty_memory_token_ids=empty_passages_token_ids,
        footer_token_ids=footer_token_ids,
    )


def reverse_ranked_chunk_ids(hits: Sequence[SearchHit]) -> list[str]:
    """Unique retrieved chunk ids with the best-ranked chunk last.

    Matches the LoCoMo injection order convention (best-scoring content sits
    closest to the question).
    """
    unique_ids = list(dict.fromkeys(str(hit.id) for hit in hits))
    return list(reversed(unique_ids))


def build_rag_memory_token_ids(
    scaffold: MemoryScaffoldTokens,
    ordered_chunk_token_ids: Sequence[Sequence[int]],
) -> list[int]:
    token_ids = list(scaffold.header_token_ids)
    if ordered_chunk_token_ids:
        for chunk_token_ids in ordered_chunk_token_ids:
            token_ids.extend(chunk_token_ids)
    else:
        token_ids.extend(scaffold.empty_memory_token_ids)
    token_ids.extend(scaffold.footer_token_ids)
    return token_ids


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
) -> KvQueryTokens:
    query_token_ids = tokenize_messages(
        tokenizer,
        build_rag_query_messages(query, answer_instruction=answer_instruction),
    )
    query_token_ids, stripped_query_bos = strip_duplicate_query_bos(
        tokenizer,
        memory_token_ids=memory_token_ids,
        query_token_ids=query_token_ids,
    )
    return KvQueryTokens(token_ids=query_token_ids, stripped_query_bos=stripped_query_bos)


def calculate_rag_memory_budget(
    *,
    query_token_count: int,
    max_position: int,
    max_model_len: int,
    max_answer_tokens: int,
) -> int:
    if max_position < 1:
        raise ValueError("max_position must be >= 1.")
    if max_model_len < 1:
        raise ValueError("max_model_len must be >= 1.")
    if max_answer_tokens < 0:
        raise ValueError("max_answer_tokens must be >= 0.")
    model_memory_budget = max_model_len - query_token_count - max_answer_tokens
    if model_memory_budget < 0:
        raise RuntimeError(
            "Query and requested answer tokens exceed kv_max_model_len: "
            f"query={query_token_count} answer={max_answer_tokens} "
            f"kv_max_model_len={max_model_len}."
        )
    return min(max_position, model_memory_budget)


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
        f"{memory_token_budget} (top_k={top_k}, chunk_size={chunk_size}). Lower --top-k "
        "or --chunk-size, or raise --kv-max-position and --kv-max-model-len."
    )


def require_identical_token_ids(precomputed_token_ids: list[int], live_token_ids: list[int]) -> None:
    """KV/prefix parity guard: cached token ids must match the live tokenizer."""
    if precomputed_token_ids == live_token_ids:
        return
    mismatch_index = next(
        (
            index
            for index, (left, right) in enumerate(zip(precomputed_token_ids, live_token_ids))
            if left != right
        ),
        min(len(precomputed_token_ids), len(live_token_ids)),
    )
    raise RuntimeError(
        "Precomputed memory tokens differ from the live tokenizer at "
        f"index={mismatch_index}: precomputed_length={len(precomputed_token_ids)} "
        f"live_length={len(live_token_ids)}."
    )
