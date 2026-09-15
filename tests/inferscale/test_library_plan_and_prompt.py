from __future__ import annotations

import pytest

from inferscale.v1.kv.plan import build_encoding_plan
from inferscale.v1.kv.prompt import (
    SCAFFOLD_PLACEHOLDER,
    block_aligned_prefix,
    build_memory_token_ids,
    build_prompt_tokens,
    build_query_tokens,
    build_scaffold_tokens,
    memory_token_budget,
    require_identical_token_ids,
    require_memory_within_budget,
)
from inferscale.v1.kv.tokenization import encode_text_no_special
from library_fakes import FakeTokenizer


def test_encoding_plan_keeps_target_and_truncates_oldest_context() -> None:
    plan = build_encoding_plan(
        "c1",
        [7, 8, 9],
        context_token_ids=[1, 2, 3, 4, 5],
        context_ids=("t1", "t2"),
        max_input_tokens=6,
    )
    assert plan.chunk_id == "c1"
    assert plan.context_token_ids == [3, 4, 5]
    assert plan.input_token_ids == [3, 4, 5, 7, 8, 9]
    assert (plan.slice_start, plan.slice_end) == (3, 6)
    assert plan.raw_context_tokens == 5
    assert plan.context_truncated_tokens == 2
    assert plan.context_ids == ("t1", "t2")


def test_encoding_plan_without_context() -> None:
    plan = build_encoding_plan("c", [4, 5], max_input_tokens=8)
    assert plan.context_token_ids == []
    assert plan.input_token_ids == [4, 5]
    assert (plan.slice_start, plan.slice_end) == (0, 2)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target_token_ids": [], "max_input_tokens": 4}, "zero tokens"),
        ({"target_token_ids": [1, 2, 3, 4, 5], "max_input_tokens": 4}, "exceeding max_position"),
        ({"target_token_ids": [1], "max_input_tokens": 0}, "max_input_tokens"),
    ],
)
def test_encoding_plan_rejects_bad_inputs(kwargs, message) -> None:
    with pytest.raises((RuntimeError, ValueError), match=message):
        build_encoding_plan("c", kwargs["target_token_ids"], max_input_tokens=kwargs["max_input_tokens"])


def test_scaffold_splits_the_template_and_pads_the_footer() -> None:
    tokenizer = FakeTokenizer()
    scaffold = build_scaffold_tokens(
        tokenizer,
        system_prompt="Use the context.\n\n",
        empty_text="(none)\n",
        block_size=16,
    )
    templated = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Use the context.\n\n" + SCAFFOLD_PLACEHOLDER}],
        tokenize=False,
        add_generation_prompt=False,
    )
    header_text, footer_text = templated.split(SCAFFOLD_PLACEHOLDER, 1)
    assert scaffold.header_token_ids == encode_text_no_special(tokenizer, header_text)
    assert scaffold.empty_token_ids == encode_text_no_special(tokenizer, "(none)\n")
    newline = encode_text_no_special(tokenizer, "\n")
    close = encode_text_no_special(tokenizer, footer_text)
    assert scaffold.footer_token_ids == newline * (15 - len(close)) + close
    assert len(scaffold.footer_token_ids) >= 15


def test_scaffold_placeholder_text_does_not_change_token_ids() -> None:
    tokenizer = FakeTokenizer()
    default = build_scaffold_tokens(tokenizer, system_prompt="S\n", empty_text="E\n")
    custom = build_scaffold_tokens(
        tokenizer, system_prompt="S\n", empty_text="E\n", placeholder="<<<OTHER_MARKER>>>"
    )
    assert default == custom


def test_scaffold_rejects_empty_prompt_and_bad_block_size() -> None:
    with pytest.raises(ValueError):
        build_scaffold_tokens(FakeTokenizer(), system_prompt="", empty_text="E")
    with pytest.raises(ValueError):
        build_scaffold_tokens(FakeTokenizer(), system_prompt="S", empty_text="E", block_size=0)


def test_query_tokens_strip_a_duplicate_bos_and_assemble_the_prompt() -> None:
    tokenizer = FakeTokenizer()
    memory = [tokenizer.bos_token_id, 50, 51]
    query = build_query_tokens(tokenizer, memory, [{"role": "user", "content": "Q?"}])
    assert query.stripped_query_bos is True
    assert query.token_ids[0] != tokenizer.bos_token_id
    prompt = build_prompt_tokens(memory, query)
    assert prompt.prompt_token_ids == memory + query.token_ids
    assert prompt.memory_token_ids == memory

    no_bos_memory = [50, 51]
    query = build_query_tokens(tokenizer, no_bos_memory, [{"role": "user", "content": "Q?"}])
    assert query.stripped_query_bos is False
    assert query.token_ids[0] == tokenizer.bos_token_id


def test_memory_budget_and_alignment_guards() -> None:
    assert memory_token_budget(query_token_count=10, max_position=100, max_model_len=200, max_answer_tokens=20) == 100
    assert memory_token_budget(query_token_count=10, max_position=1000, max_model_len=200, max_answer_tokens=20) == 170
    with pytest.raises(RuntimeError, match="exceed max_model_len"):
        memory_token_budget(query_token_count=190, max_position=100, max_model_len=200, max_answer_tokens=20)
    require_memory_within_budget(100, 100)
    with pytest.raises(RuntimeError, match="memory budget"):
        require_memory_within_budget(101, 100)

    assert block_aligned_prefix(40, footer_token_count=15, block_size=16) == (32, 8)
    with pytest.raises(RuntimeError, match="recomputed tail"):
        block_aligned_prefix(40, footer_token_count=2, block_size=16)


def test_memory_token_ids_and_parity_guard() -> None:
    scaffold = build_scaffold_tokens(FakeTokenizer(), system_prompt="S\n", empty_text="E\n", block_size=4)
    with_chunks = build_memory_token_ids(scaffold, [[9, 9], [8]])
    assert with_chunks == scaffold.header_token_ids + [9, 9, 8] + scaffold.footer_token_ids
    empty = build_memory_token_ids(scaffold, [])
    assert empty == scaffold.header_token_ids + scaffold.empty_token_ids + scaffold.footer_token_ids
    require_identical_token_ids([1, 2], [1, 2])
    with pytest.raises(RuntimeError, match="index=1"):
        require_identical_token_ids([1, 2], [1, 3])
