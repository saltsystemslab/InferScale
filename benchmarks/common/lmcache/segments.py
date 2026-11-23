from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from inferscale.v1.types import ScaffoldTokens


@dataclass(slots=True, frozen=True)
class BlendMemoryLayout:
    token_ids: list[int]
    separator_token_ids: list[int]
    segment_count: int
    expected_matched_tokens: int
    scaffold_tokens: int


def separator_token_ids(tokenizer: Any, special_str: str) -> list[int]:
    token_ids = list(tokenizer.encode(special_str))[1:]
    if not token_ids:
        raise RuntimeError(f"The blend separator {special_str!r} encodes to no tokens.")
    return token_ids


def blend_memory_layout(
    scaffold: ScaffoldTokens,
    ordered_chunk_token_ids: Sequence[Sequence[int]],
    separator: Sequence[int],
) -> BlendMemoryLayout:
    separator_ids = list(separator)
    if not separator_ids:
        raise ValueError("The blend separator must be non-empty.")
    chunks = [list(chunk) for chunk in ordered_chunk_token_ids]
    if not chunks:
        return _empty_layout(scaffold, separator_ids)
    token_ids = list(scaffold.header_token_ids)
    for chunk in chunks:
        require_chunk_without_separator(chunk, separator_ids)
        token_ids.extend(separator_ids)
        token_ids.extend(chunk)
    tokens_through_last_chunk = len(token_ids)
    token_ids.extend(separator_ids)
    token_ids.extend(scaffold.footer_token_ids)
    chunk_tokens = sum(len(chunk) for chunk in chunks)
    return BlendMemoryLayout(
        token_ids=token_ids,
        separator_token_ids=separator_ids,
        segment_count=1 + len(chunks),
        expected_matched_tokens=tokens_through_last_chunk,
        scaffold_tokens=len(token_ids) - chunk_tokens,
    )


def _empty_layout(scaffold: ScaffoldTokens, separator_ids: list[int]) -> BlendMemoryLayout:
    token_ids = [
        *scaffold.header_token_ids,
        *scaffold.empty_token_ids,
        *scaffold.footer_token_ids,
    ]
    return BlendMemoryLayout(
        token_ids=token_ids,
        separator_token_ids=separator_ids,
        segment_count=1,
        expected_matched_tokens=0,
        scaffold_tokens=len(token_ids),
    )


def warm_up_prompts(
    header_token_ids: Sequence[int],
    chunk_token_ids: Iterable[Sequence[int]],
    separator: Sequence[int],
) -> list[list[int]]:
    unique_prompts: dict[tuple[int, ...], None] = {tuple(header_token_ids): None}
    for chunk in chunk_token_ids:
        require_chunk_without_separator(chunk, separator)
        unique_prompts.setdefault(tuple(chunk), None)
    return [list(prompt) for prompt in unique_prompts]


def require_chunk_without_separator(chunk: Sequence[int], separator: Sequence[int]) -> None:
    if not chunk:
        raise ValueError("Blend chunks must hold at least one token.")
    if contains_separator(chunk, separator):
        raise ValueError(
            "A blend chunk contains the separator token sequence; LMCache would split it "
            "into segments that never match its warm-up. Choose another blend_special_str."
        )


def contains_separator(token_ids: Sequence[int], separator: Sequence[int]) -> bool:
    width = len(separator)
    if width == 0 or len(token_ids) < width:
        return False
    target = list(separator)
    return any(
        list(token_ids[start : start + width]) == target
        for start in range(len(token_ids) - width + 1)
    )


def prompt_token_count(prompts: Sequence[Sequence[int]]) -> int:
    return sum(len(prompt) for prompt in prompts)
