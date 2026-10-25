from __future__ import annotations

import pytest
from library_fakes import FakeTokenizer

from benchmarks.common.lmcache.segments import (
    blend_memory_layout,
    contains_separator,
    prompt_token_count,
    separator_token_ids,
    warm_up_prompts,
)
from inferscale.v1.kv.prompt import build_scaffold_tokens
from inferscale.v1.types import ScaffoldTokens

SEPARATOR = " # # "


def _scaffold() -> ScaffoldTokens:
    return build_scaffold_tokens(
        FakeTokenizer(), system_prompt="Remember:\n", empty_text="(none)\n", block_size=16
    )


def _split(token_ids: list[int], separator: list[int]) -> list[list[int]]:
    """The segments between separator occurrences, the way LMCache splits a prompt."""
    segments: list[list[int]] = []
    start = index = 0
    width = len(separator)
    while index <= len(token_ids) - width:
        if token_ids[index : index + width] == separator:
            segments.append(token_ids[start:index])
            index += width
            start = index
        else:
            index += 1
    segments.append(token_ids[start:])
    return segments


def test_separator_mirrors_lmcache_derivation() -> None:
    tokenizer = FakeTokenizer()

    assert separator_token_ids(tokenizer, SEPARATOR) == tokenizer.encode(SEPARATOR)[1:]


def test_separator_of_a_tokenizer_without_bos_loses_its_first_token() -> None:
    class NoBosTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

    # LMCache always drops the first token, so the prompts must do the same.
    assert separator_token_ids(NoBosTokenizer(), "##") == [ord("#")]


def test_separator_that_encodes_to_one_token_is_rejected() -> None:
    class OneTokenTokenizer:
        def encode(self, text: str) -> list[int]:
            return [7]

    with pytest.raises(RuntimeError, match="encodes to no tokens"):
        separator_token_ids(OneTokenTokenizer(), "#")


def test_layout_is_header_separated_chunks_and_footer() -> None:
    scaffold = _scaffold()
    separator = separator_token_ids(FakeTokenizer(), SEPARATOR)
    chunks = [[101, 102, 103], [201, 202]]

    layout = blend_memory_layout(scaffold, chunks, separator)

    assert layout.token_ids == [
        *scaffold.header_token_ids,
        *separator,
        101,
        102,
        103,
        *separator,
        201,
        202,
        *separator,
        *scaffold.footer_token_ids,
    ]
    assert layout.separator_token_ids == separator
    assert layout.segment_count == 3
    assert layout.expected_matched_tokens == len(scaffold.header_token_ids) + 2 * len(separator) + 5
    assert layout.scaffold_tokens == len(layout.token_ids) - 5


def test_segments_between_separators_are_exactly_the_warm_up_prompts() -> None:
    scaffold = _scaffold()
    separator = separator_token_ids(FakeTokenizer(), SEPARATOR)
    chunks = [[101, 102, 103], [201, 202]]

    layout = blend_memory_layout(scaffold, chunks, separator)
    prompts = warm_up_prompts(scaffold.header_token_ids, chunks, separator)
    segments = _split(layout.token_ids, separator)

    assert segments[:-1] == prompts
    assert segments[-1] == scaffold.footer_token_ids
    # LMCache matches up to the end of the last chunk; the final separator and footer are new.
    assert layout.expected_matched_tokens == len(layout.token_ids) - len(separator) - len(
        scaffold.footer_token_ids
    )


def test_warm_up_prompts_deduplicate_chunks_and_keep_their_order() -> None:
    assert warm_up_prompts([1, 2], [[5], [6, 7], [5]], [9, 9]) == [[1, 2], [5], [6, 7]]


def test_empty_layout_is_prompt_injection_memory_with_nothing_to_match() -> None:
    scaffold = _scaffold()

    layout = blend_memory_layout(scaffold, [], [9, 9])

    assert layout.token_ids == [
        *scaffold.header_token_ids,
        *scaffold.empty_token_ids,
        *scaffold.footer_token_ids,
    ]
    assert (layout.segment_count, layout.expected_matched_tokens) == (1, 0)
    assert layout.scaffold_tokens == len(layout.token_ids)


@pytest.mark.parametrize("chunk", [[], [3, 9, 9, 4]])
def test_empty_chunks_and_chunks_holding_the_separator_are_rejected(chunk: list[int]) -> None:
    with pytest.raises(ValueError):
        blend_memory_layout(_scaffold(), [chunk], [9, 9])
    with pytest.raises(ValueError):
        warm_up_prompts([1], [chunk], [9, 9])


def test_empty_separator_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        blend_memory_layout(_scaffold(), [[1]], [])


def test_separator_search_and_token_count() -> None:
    assert contains_separator([1, 9, 9, 2], [9, 9])
    assert not contains_separator([1, 9, 2, 9], [9, 9])
    assert not contains_separator([9], [9, 9])
    assert prompt_token_count([[1, 2], [3]]) == 3
