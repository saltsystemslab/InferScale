from __future__ import annotations

from collections.abc import Sequence
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

from .context import plan_prompt_memory
from .data import QuestionAnswer
from .facts import unique_memory_facts
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
    token_ids: list[int]
    fact_plan: MemoryFactPlan


@dataclass(slots=True, frozen=True)
class MemoryQuery:
    scaffold: ScaffoldTokens
    query_tokens: QueryTokens
    memory_token_budget: int


@dataclass(slots=True, frozen=True)
class SelectedMemoryFacts:
    retrieved: list[RetrievedFact]
    selected: list[RetrievedFact]
    token_ids: dict[str, list[int]]

    def ordered_token_ids(self) -> list[list[int]]:
        return [self.token_ids[fact.memory_id] for fact in self.selected]


def format_memory_fact(fact: RetrievedFact) -> str:
    return fact.text.strip() + "\n"


def select_memory_facts(tokenizer: Any, hits: Sequence[SearchHit]) -> SelectedMemoryFacts:
    retrieved = unique_memory_facts(hits)
    selected = list(reversed(retrieved))
    token_ids = {
        fact.memory_id: encode_text_no_special(tokenizer, format_memory_fact(fact))
        for fact in selected
    }
    return SelectedMemoryFacts(retrieved=retrieved, selected=selected, token_ids=token_ids)


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
    hits: list[SearchHit],
    *,
    memory_token_budget: int | None = None,
    memory_scaffold: ScaffoldTokens | None = None,
) -> MemoryPromptTokens:
    facts = select_memory_facts(tokenizer, hits)
    scaffold = memory_scaffold or extract_memory_scaffold_token_ids(tokenizer)
    heading_token_ids = [] if facts.selected else scaffold.empty_token_ids
    scaffold_token_count = (
        len(scaffold.header_token_ids) + len(heading_token_ids) + len(scaffold.footer_token_ids)
    )
    fact_stream = [token for token_ids in facts.ordered_token_ids() for token in token_ids]
    if memory_token_budget is None:
        memory_token_budget = scaffold_token_count + len(fact_stream)
    fact_plan = plan_prompt_memory(
        facts.selected,
        retrieved_facts=facts.retrieved,
        memory_token_budget=memory_token_budget,
        scaffold_token_count=scaffold_token_count,
        fact_token_ids=facts.token_ids,
    )
    token_ids = [
        *scaffold.header_token_ids,
        *heading_token_ids,
        *fact_stream,
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
    query_tokens = build_query_tokens_for_memory(tokenizer, memory_token_ids, qa)
    return build_prompt_tokens(memory_token_ids, query_tokens)
