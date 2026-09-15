"""LoCoMo prompt layout: the memory scaffold, rendered facts, and the question turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inferscale.v1.kv.prompt import (
    PromptTokens,
    QueryTokens,
    ScaffoldTokens,
    build_prompt_tokens,
    build_query_tokens,
    build_scaffold_tokens,
    memory_token_budget,
)
from inferscale.v1.kv.tokenization import encode_text_no_special
from inferscale.v1.types import SearchHit

from .context import (
    build_memory_fact_plan,
    context_turn_token_ids,
    previous_turn_context_turns,
    reverse_ranked_memory_facts,
    unique_memory_facts,
)
from .data import ConversationSample, QuestionAnswer
from .types import MemoryFact, MemoryFactPlan

MEMORY_TEMPLATE_PLACEHOLDER = "<<<LOCOMO_JASPER_MEMORY_GOES_HERE>>>"
MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant that remembers details from past conversations. "
    "Answer questions based on the conversation history provided. "
    "The following is a conversation history between two people:\n\n"
)
EMPTY_MEMORY_TEXT = "(No relevant memories found)\n"


@dataclass(slots=True, frozen=True)
class MemoryPromptTokens:
    token_ids: list[int]
    selected_fact_ids: list[str]
    fact_plan: MemoryFactPlan


def format_memory_fact(fact: MemoryFact) -> str:
    return fact.text.strip() + "\n"


def extract_memory_scaffold_token_ids(tokenizer: Any, *, block_size: int = 16) -> ScaffoldTokens:
    return build_scaffold_tokens(
        tokenizer,
        system_prompt=MEMORY_SYSTEM_PROMPT,
        empty_text=EMPTY_MEMORY_TEXT,
        block_size=block_size,
        placeholder=MEMORY_TEMPLATE_PLACEHOLDER,
    )


def build_memory_prompt_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    hits: list[SearchHit],
    *,
    context_window: int = 0,
    memory_token_budget: int | None = None,
    memory_scaffold: ScaffoldTokens | None = None,
    render_context_turns: bool = False,
) -> MemoryPromptTokens:
    retrieved_facts = unique_memory_facts(hits)
    selected_facts = reverse_ranked_memory_facts(hits)
    scaffold = memory_scaffold or extract_memory_scaffold_token_ids(tokenizer)
    memory_heading_token_ids = [] if selected_facts else scaffold.empty_token_ids
    scaffold_token_count = (
        len(scaffold.header_token_ids)
        + len(memory_heading_token_ids)
        + len(scaffold.footer_token_ids)
    )
    fact_token_ids = {
        fact.memory_id: encode_text_no_special(tokenizer, format_memory_fact(fact))
        for fact in selected_facts
    }
    memory_stream = _memory_stream_token_ids(
        tokenizer,
        sample,
        selected_facts,
        fact_token_ids=fact_token_ids,
        context_window=context_window if render_context_turns else 0,
    )
    if memory_token_budget is None:
        memory_token_budget = (
            scaffold_token_count
            + memory_stream.context_text_tokens
            + sum(len(token_ids) for token_ids in fact_token_ids.values())
        )
    fact_plan = build_memory_fact_plan(
        selected_facts,
        retrieved_facts=retrieved_facts,
        context_window=context_window,
        memory_token_budget=memory_token_budget,
        scaffold_token_count=scaffold_token_count,
        fact_token_ids=fact_token_ids,
        context_turn_ids=memory_stream.context_turn_ids,
        context_text_tokens=memory_stream.context_text_tokens,
    )

    token_ids = list(scaffold.header_token_ids)
    token_ids.extend(memory_heading_token_ids)
    token_ids.extend(memory_stream.token_ids)
    token_ids.extend(scaffold.footer_token_ids)
    if len(token_ids) != fact_plan.memory_tokens:
        raise AssertionError(
            "Planned memory token count does not match the composed prefix prompt."
        )
    return MemoryPromptTokens(
        token_ids=token_ids,
        selected_fact_ids=list(fact_plan.injected_fact_ids),
        fact_plan=fact_plan,
    )


@dataclass(slots=True, frozen=True)
class _MemoryStreamTokens:
    token_ids: list[int]
    context_turn_ids: tuple[str, ...]
    context_text_tokens: int


def _memory_stream_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    selected_facts: list[MemoryFact],
    *,
    fact_token_ids: dict[str, list[int]],
    context_window: int,
) -> _MemoryStreamTokens:
    """Render reverse-ranked facts, optionally preceded by their context turns."""
    if context_window <= 0 or not selected_facts:
        token_ids: list[int] = []
        for fact in selected_facts:
            token_ids.extend(fact_token_ids[fact.memory_id])
        return _MemoryStreamTokens(token_ids=token_ids, context_turn_ids=(), context_text_tokens=0)

    turn_token_cache: dict[str, list[int]] = {}
    seen_context_turn_ids: set[str] = set()
    token_ids: list[int] = []
    context_turn_ids: list[str] = []
    context_text_tokens = 0
    for fact in selected_facts:
        context_turns = previous_turn_context_turns(
            sample,
            fact.source_turn_id,
            context_window,
        )
        for turn in context_turns:
            if turn.id in seen_context_turn_ids:
                continue
            seen_context_turn_ids.add(turn.id)
            turn_tokens = context_turn_token_ids(tokenizer, turn, turn_token_cache)
            token_ids.extend(turn_tokens)
            context_turn_ids.append(turn.id)
            context_text_tokens += len(turn_tokens)
        token_ids.extend(fact_token_ids[fact.memory_id])
    return _MemoryStreamTokens(
        token_ids=token_ids,
        context_turn_ids=tuple(context_turn_ids),
        context_text_tokens=context_text_tokens,
    )


def calculate_memory_token_budget(
    tokenizer: Any,
    sample: ConversationSample,
    qa: QuestionAnswer,
    *,
    memory_prefix_token_ids: list[int],
    max_position: int,
    max_model_len: int,
    max_answer_tokens: int,
    query_tokens: QueryTokens | None = None,
) -> int:
    if query_tokens is None:
        query_tokens = build_kv_query_tokens_for_memory(tokenizer, memory_prefix_token_ids, sample, qa)
    return memory_token_budget(
        query_token_count=len(query_tokens.token_ids),
        max_position=max_position,
        max_model_len=max_model_len,
        max_answer_tokens=max_answer_tokens,
    )


def build_kv_query_messages(
    sample: ConversationSample,
    qa: QuestionAnswer,
) -> list[dict[str, str]]:
    del sample
    return [
        {
            "role": "user",
            "content": (
                "Based on the conversation above, answer concisely.\n"
                f"Question: {qa.question}"
            ),
        }
    ]


def build_kv_query_tokens_for_memory(
    tokenizer: Any,
    memory_token_ids: list[int],
    sample: ConversationSample,
    qa: QuestionAnswer,
) -> QueryTokens:
    return build_query_tokens(tokenizer, memory_token_ids, build_kv_query_messages(sample, qa))


def build_kv_equivalence_prompt_token_ids(
    tokenizer: Any,
    memory_token_ids: list[int],
    sample: ConversationSample,
    qa: QuestionAnswer,
) -> PromptTokens:
    query_tokens = build_kv_query_tokens_for_memory(tokenizer, memory_token_ids, sample, qa)
    return build_prompt_tokens(memory_token_ids, query_tokens)
