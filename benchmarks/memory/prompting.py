"""LoCoMo prompt layout: the memory scaffold, rendered facts, and the question turn.

KV injection and prompt injection build byte-identical token streams from the
same pieces: a scaffold (system prompt header, an empty-memory marker when
nothing was retrieved, footer), the retrieved facts in reverse rank order,
and the question as the final chat turn.
"""

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

from .context import context_turn_token_ids, plan_prompt_memory, previous_turn_context_turns
from .data import ConversationSample, QuestionAnswer
from .facts import reverse_ranked_memory_facts, unique_memory_facts
from .types import MemoryFactPlan, RetrievedFact

MEMORY_TEMPLATE_PLACEHOLDER = "<<<LOCOMO_JASPER_MEMORY_GOES_HERE>>>"
MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant that remembers details from past conversations. "
    "Answer questions based on the conversation history provided. "
    "The following is a conversation history between two people:\n\n"
)
EMPTY_MEMORY_TEXT = "(No relevant memories found)\n"


@dataclass(slots=True, frozen=True)
class MemoryPromptTokens:
    """The memory rendered as prompt text, with the plan that accounts for it."""

    token_ids: list[int]
    fact_plan: MemoryFactPlan


@dataclass(slots=True, frozen=True)
class MemoryQuery:
    """The question side of one answer: scaffold, question tokens, and the memory budget."""

    scaffold: ScaffoldTokens
    query_tokens: QueryTokens
    memory_token_budget: int


@dataclass(slots=True, frozen=True)
class _MemoryStream:
    token_ids: list[int]
    context_turn_ids: tuple[str, ...]
    context_text_tokens: int


def format_memory_fact(fact: RetrievedFact) -> str:
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
    """The memory as prompt text; ``None`` for the budget means the memory is not bounded."""
    retrieved_facts = unique_memory_facts(hits)
    selected_facts = reverse_ranked_memory_facts(hits)
    scaffold = memory_scaffold or extract_memory_scaffold_token_ids(tokenizer)
    heading_token_ids = [] if selected_facts else scaffold.empty_token_ids
    scaffold_token_count = (
        len(scaffold.header_token_ids) + len(heading_token_ids) + len(scaffold.footer_token_ids)
    )
    fact_token_ids = {
        fact.memory_id: encode_text_no_special(tokenizer, format_memory_fact(fact))
        for fact in selected_facts
    }
    if render_context_turns and context_window > 0:
        stream = _facts_with_context_stream(
            tokenizer, sample, selected_facts, fact_token_ids, context_window
        )
    else:
        stream = _facts_only_stream(selected_facts, fact_token_ids)
    if memory_token_budget is None:
        memory_token_budget = (
            scaffold_token_count
            + stream.context_text_tokens
            + sum(len(token_ids) for token_ids in fact_token_ids.values())
        )
    fact_plan = plan_prompt_memory(
        selected_facts,
        retrieved_facts=retrieved_facts,
        context_window=context_window,
        memory_token_budget=memory_token_budget,
        scaffold_token_count=scaffold_token_count,
        fact_token_ids=fact_token_ids,
        context_turn_ids=stream.context_turn_ids,
        context_text_tokens=stream.context_text_tokens,
    )
    token_ids = [
        *scaffold.header_token_ids,
        *heading_token_ids,
        *stream.token_ids,
        *scaffold.footer_token_ids,
    ]
    if len(token_ids) != fact_plan.memory_tokens:
        raise AssertionError("Planned memory token count does not match the rendered memory.")
    return MemoryPromptTokens(token_ids=token_ids, fact_plan=fact_plan)


def memory_token_budget_for(
    query_tokens: QueryTokens, *, max_position: int, max_model_len: int, max_answer_tokens: int
) -> int:
    return memory_token_budget(
        query_token_count=len(query_tokens.token_ids),
        max_position=max_position,
        max_model_len=max_model_len,
        max_answer_tokens=max_answer_tokens,
    )


def build_query_messages(qa: QuestionAnswer) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"Based on the conversation above, answer concisely.\nQuestion: {qa.question}"
            ),
        }
    ]


def build_query_tokens_for_memory(
    tokenizer: Any, memory_token_ids: list[int], qa: QuestionAnswer
) -> QueryTokens:
    return build_query_tokens(tokenizer, memory_token_ids, build_query_messages(qa))


def build_memory_query(
    tokenizer: Any,
    qa: QuestionAnswer,
    *,
    block_size: int,
    max_position: int,
    max_model_len: int,
    max_answer_tokens: int,
) -> MemoryQuery:
    """The scaffold, question tokens, and memory budget both answer clients start from."""
    scaffold = extract_memory_scaffold_token_ids(tokenizer, block_size=block_size)
    query_tokens = build_query_tokens_for_memory(tokenizer, scaffold.header_token_ids, qa)
    budget = memory_token_budget_for(
        query_tokens,
        max_position=max_position,
        max_model_len=max_model_len,
        max_answer_tokens=max_answer_tokens,
    )
    return MemoryQuery(scaffold=scaffold, query_tokens=query_tokens, memory_token_budget=budget)


def build_kv_equivalence_prompt_token_ids(
    tokenizer: Any, memory_token_ids: list[int], qa: QuestionAnswer
) -> PromptTokens:
    """The full prompt a prompt-injection request sends for the given memory tokens."""
    query_tokens = build_query_tokens_for_memory(tokenizer, memory_token_ids, qa)
    return build_prompt_tokens(memory_token_ids, query_tokens)


def _facts_only_stream(
    selected_facts: list[RetrievedFact], fact_token_ids: dict[str, list[int]]
) -> _MemoryStream:
    token_ids = [token for fact in selected_facts for token in fact_token_ids[fact.memory_id]]
    return _MemoryStream(token_ids=token_ids, context_turn_ids=(), context_text_tokens=0)


def _facts_with_context_stream(
    tokenizer: Any,
    sample: ConversationSample,
    selected_facts: list[RetrievedFact],
    fact_token_ids: dict[str, list[int]],
    context_window: int,
) -> _MemoryStream:
    """Each fact preceded by its context turns; a turn is rendered once, before its first fact."""
    turn_token_cache: dict[str, list[int]] = {}
    token_ids: list[int] = []
    context_turn_ids: list[str] = []
    context_text_tokens = 0
    for fact in selected_facts:
        for turn in previous_turn_context_turns(sample, fact.source_turn_id, context_window):
            if turn.id in context_turn_ids:
                continue
            turn_tokens = context_turn_token_ids(tokenizer, turn, turn_token_cache)
            token_ids.extend(turn_tokens)
            context_turn_ids.append(turn.id)
            context_text_tokens += len(turn_tokens)
        token_ids.extend(fact_token_ids[fact.memory_id])
    return _MemoryStream(
        token_ids=token_ids,
        context_turn_ids=tuple(context_turn_ids),
        context_text_tokens=context_text_tokens,
    )
