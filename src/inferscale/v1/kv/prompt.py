"""Prompt scaffold and query tokens around an injected KV memory.

The prompt the engine sees is ``header + chunks (or empty text) + footer +
query``. The header and footer come from the chat template of one system
message; the footer is padded at the token level with newline tokens so the
block-aligned KV prefix always covers every injected chunk token and only
footer padding can fall into the recomputed tail.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import PromptTokens, QueryTokens, ScaffoldTokens
from .tokenization import encode_text_no_special

SCAFFOLD_PLACEHOLDER = "<<<INFERSCALE_CONTEXT_GOES_HERE>>>"


def build_scaffold_tokens(
    tokenizer: Any,
    *,
    system_prompt: str,
    empty_text: str,
    block_size: int = 16,
    placeholder: str = SCAFFOLD_PLACEHOLDER,
) -> ScaffoldTokens:
    """Chat-templated scaffold split around the context placeholder.

    The footer is padded at the token level with newline tokens to at least
    ``block_size - 1`` tokens. Text-level padding cannot guarantee that: chat
    templates trim message content and BPE tokenizers merge whitespace runs.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if not system_prompt:
        raise ValueError("system_prompt must be non-empty.")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        templated = apply_chat_template_non_thinking(
            tokenizer,
            [{"role": "system", "content": system_prompt + placeholder}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        templated = f"SYSTEM: {system_prompt}{placeholder}"

    if placeholder not in templated:
        raise RuntimeError("The chat template removed the context placeholder.")
    header_text, footer_text = templated.split(placeholder, 1)
    header_token_ids = encode_text_no_special(tokenizer, header_text)
    empty_token_ids = encode_text_no_special(tokenizer, empty_text)
    footer_close_token_ids = encode_text_no_special(tokenizer, footer_text)
    newline_token_ids = encode_text_no_special(tokenizer, "\n")
    if not newline_token_ids:
        raise RuntimeError("Tokenizer produced no tokens for a newline pad.")
    pad_repeats = max(0, block_size - 1 - len(footer_close_token_ids))
    footer_token_ids = newline_token_ids * pad_repeats + footer_close_token_ids
    if not header_token_ids or not empty_token_ids:
        raise RuntimeError(
            f"Empty scaffold tokens: header={len(header_token_ids)} "
            f"empty={len(empty_token_ids)} footer={len(footer_token_ids)}."
        )
    if len(footer_token_ids) < block_size - 1:
        raise RuntimeError(
            "The scaffold does not contain enough trailing whitespace to keep "
            f"retrieved chunks outside the recomputed KV tail: {len(footer_token_ids)} "
            f"footer token(s); need at least {block_size - 1}."
        )
    return ScaffoldTokens(
        header_token_ids=header_token_ids,
        empty_token_ids=empty_token_ids,
        footer_token_ids=footer_token_ids,
    )


def build_memory_token_ids(
    scaffold: ScaffoldTokens,
    ordered_chunk_token_ids: Sequence[Sequence[int]],
) -> list[int]:
    """Token ids of the memory section as the prompt-prefix baseline renders it."""
    token_ids = list(scaffold.header_token_ids)
    if ordered_chunk_token_ids:
        for chunk_token_ids in ordered_chunk_token_ids:
            token_ids.extend(chunk_token_ids)
    else:
        token_ids.extend(scaffold.empty_token_ids)
    token_ids.extend(scaffold.footer_token_ids)
    return token_ids


def build_query_tokens(
    tokenizer: Any,
    memory_token_ids: Sequence[int],
    messages: list[dict[str, str]],
) -> QueryTokens:
    query_token_ids = tokenize_messages(tokenizer, messages)
    query_token_ids, stripped_query_bos = strip_duplicate_query_bos(
        tokenizer,
        memory_token_ids=list(memory_token_ids),
        query_token_ids=query_token_ids,
    )
    return QueryTokens(token_ids=query_token_ids, stripped_query_bos=stripped_query_bos)


def build_prompt_tokens(
    memory_token_ids: Sequence[int],
    query_tokens: QueryTokens,
) -> PromptTokens:
    return PromptTokens(
        memory_token_ids=list(memory_token_ids),
        query_token_ids=list(query_tokens.token_ids),
        prompt_token_ids=list(memory_token_ids) + list(query_tokens.token_ids),
        stripped_query_bos=query_tokens.stripped_query_bos,
    )


def memory_token_budget(
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
            "Query and requested answer tokens exceed max_model_len: "
            f"query={query_token_count} answer={max_answer_tokens} "
            f"max_model_len={max_model_len}."
        )
    return min(max_position, model_memory_budget)


def require_memory_within_budget(memory_token_count: int, memory_token_budget: int) -> None:
    """Strict fail on over-budget composition; retrieved chunks are never dropped."""
    if memory_token_count <= memory_token_budget:
        return
    raise RuntimeError(
        f"Composed memory needs {memory_token_count} tokens but the memory budget is "
        f"{memory_token_budget}. Retrieve fewer or shorter chunks, or raise "
        "max_position and max_model_len."
    )


def block_aligned_prefix(
    memory_token_count: int,
    footer_token_count: int,
    block_size: int,
) -> tuple[int, int]:
    """Split the memory into the block-aligned injected prefix and the recomputed tail.

    Every injected chunk token must sit inside the aligned prefix; only footer
    padding may fall into the tail the engine recomputes.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    loaded_memory_tokens = (memory_token_count // block_size) * block_size
    chunk_tokens_end = memory_token_count - footer_token_count
    if loaded_memory_tokens < chunk_tokens_end:
        raise RuntimeError(
            "The block-aligned KV prefix would leave retrieved chunk tokens in the "
            f"recomputed tail: loaded={loaded_memory_tokens} chunks_end={chunk_tokens_end} "
            f"memory={memory_token_count} block_size={block_size}."
        )
    return loaded_memory_tokens, memory_token_count - loaded_memory_tokens


def require_identical_token_ids(precomputed_token_ids: list[int], live_token_ids: list[int]) -> None:
    """Injected token ids must match what the live tokenizer would produce."""
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


def strip_duplicate_query_bos(
    tokenizer: Any,
    *,
    memory_token_ids: list[int],
    query_token_ids: list[int],
) -> tuple[list[int], bool]:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if (
        bos_token_id is not None
        and memory_token_ids
        and query_token_ids
        and memory_token_ids[0] == bos_token_id
        and query_token_ids[0] == bos_token_id
    ):
        return list(query_token_ids[1:]), True
    return list(query_token_ids), False


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return list(
            apply_chat_template_non_thinking(
                tokenizer,
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

    text = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Tokenizer has neither apply_chat_template nor encode.")
    return list(encode(text))


def apply_chat_template_non_thinking(
    tokenizer: Any,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> Any:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError("Tokenizer has no apply_chat_template method.")

    try:
        return apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return apply_chat_template(messages, **kwargs)
