from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data import ConversationSample, QuestionAnswer
from ..vector_types import SearchHit
from .context import (
    build_memory_fact_plan,
    reverse_ranked_memory_facts,
    context_turn_token_ids,
    previous_turn_context_turns,
    unique_memory_facts,
)
from .tokenization import encode_text_no_special
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


@dataclass(slots=True, frozen=True)
class MemoryScaffoldTokens:
    header_token_ids: list[int]
    memory_list_header_token_ids: list[int]
    empty_memory_token_ids: list[int]
    footer_token_ids: list[int]


@dataclass(slots=True, frozen=True)
class KvPromptTokens:
    memory_token_ids: list[int]
    query_token_ids: list[int]
    prompt_token_ids: list[int]
    stripped_query_bos: bool


@dataclass(slots=True, frozen=True)
class KvQueryTokens:
    token_ids: list[int]
    stripped_query_bos: bool


def format_memory_fact(fact: MemoryFact) -> str:
    return fact.text.strip() + "\n"


def extract_memory_scaffold_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    *,
    block_size: int = 16,
) -> MemoryScaffoldTokens:
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    memory_placeholder = MEMORY_TEMPLATE_PLACEHOLDER + ("\n" * block_size)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        templated = apply_chat_template_non_thinking(
            tokenizer,
            [{"role": "system", "content": MEMORY_SYSTEM_PROMPT + memory_placeholder}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        templated = f"SYSTEM: {MEMORY_SYSTEM_PROMPT}{memory_placeholder}"

    if MEMORY_TEMPLATE_PLACEHOLDER not in templated:
        raise RuntimeError("Answer prompt chat template removed the memory placeholder.")
    header_text, footer_text = templated.split(MEMORY_TEMPLATE_PLACEHOLDER, 1)
    header_token_ids = encode_text_no_special(tokenizer, header_text)
    empty_memory_token_ids = encode_text_no_special(tokenizer, EMPTY_MEMORY_TEXT)
    footer_token_ids = encode_text_no_special(tokenizer, footer_text)
    if (
        not header_token_ids
        or not empty_memory_token_ids
    ):
        raise RuntimeError(
            f"Empty answer scaffold tokens: header={len(header_token_ids)} "
            f"empty_memory={len(empty_memory_token_ids)} "
            f"footer={len(footer_token_ids)}."
        )
    if len(footer_token_ids) < block_size - 1:
        raise RuntimeError(
            "The memory scaffold does not contain enough trailing whitespace to keep "
            "retrieved facts outside the recomputed KV tail."
        )
    return MemoryScaffoldTokens(
        header_token_ids=header_token_ids,
        memory_list_header_token_ids=[],
        empty_memory_token_ids=empty_memory_token_ids,
        footer_token_ids=footer_token_ids,
    )


def build_memory_prompt_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    hits: list[SearchHit],
    *,
    context_window: int = 0,
    memory_token_budget: int | None = None,
    memory_scaffold: MemoryScaffoldTokens | None = None,
    render_context_turns: bool = False,
) -> MemoryPromptTokens:
    retrieved_facts = unique_memory_facts(hits)
    selected_facts = reverse_ranked_memory_facts(hits)
    scaffold = memory_scaffold or extract_memory_scaffold_token_ids(tokenizer, sample)
    memory_heading_token_ids = (
        scaffold.memory_list_header_token_ids
        if selected_facts
        else scaffold.empty_memory_token_ids
    )
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
    query_tokens: KvQueryTokens | None = None,
    memory_scaffold: MemoryScaffoldTokens | None = None,
) -> int:
    if max_position < 1:
        raise ValueError("max_position must be >= 1.")
    if max_model_len < 1:
        raise ValueError("max_model_len must be >= 1.")
    if max_answer_tokens < 0:
        raise ValueError("max_answer_tokens must be >= 0.")

    if query_tokens is None:
        query_tokens = build_kv_query_tokens_for_memory(
            tokenizer,
            memory_prefix_token_ids,
            sample,
            qa,
            memory_scaffold=memory_scaffold,
        )
    model_memory_budget = max_model_len - len(query_tokens.token_ids) - max_answer_tokens
    if model_memory_budget < 0:
        raise RuntimeError(
            "Query and requested answer tokens exceed kv_max_model_len: "
            f"query={len(query_tokens.token_ids)} answer={max_answer_tokens} "
            f"kv_max_model_len={max_model_len}."
        )
    return min(max_position, model_memory_budget)


def build_kv_query_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    qa: QuestionAnswer,
    *,
    memory_scaffold: MemoryScaffoldTokens | None = None,
) -> list[int]:
    del memory_scaffold
    return tokenize_messages(tokenizer, build_kv_query_messages(sample, qa))


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


def build_kv_equivalence_prompt_token_ids(
    tokenizer: Any,
    memory_token_ids: list[int],
    sample: ConversationSample,
    qa: QuestionAnswer,
    *,
    memory_scaffold: MemoryScaffoldTokens | None = None,
) -> KvPromptTokens:
    query_tokens = build_kv_query_tokens_for_memory(
        tokenizer,
        memory_token_ids,
        sample,
        qa,
        memory_scaffold=memory_scaffold,
    )
    return build_kv_equivalence_prompt_from_query_tokens(
        memory_token_ids,
        query_tokens,
    )


def build_kv_query_tokens_for_memory(
    tokenizer: Any,
    memory_token_ids: list[int],
    sample: ConversationSample,
    qa: QuestionAnswer,
    *,
    memory_scaffold: MemoryScaffoldTokens | None = None,
) -> KvQueryTokens:
    query_token_ids = build_kv_query_token_ids(
        tokenizer,
        sample,
        qa,
        memory_scaffold=memory_scaffold,
    )
    query_token_ids, stripped_query_bos = strip_duplicate_query_bos(
        tokenizer,
        memory_token_ids=memory_token_ids,
        query_token_ids=query_token_ids,
    )
    return KvQueryTokens(
        token_ids=query_token_ids,
        stripped_query_bos=stripped_query_bos,
    )


def build_kv_equivalence_prompt_from_query_tokens(
    memory_token_ids: list[int],
    query_tokens: KvQueryTokens,
) -> KvPromptTokens:
    return KvPromptTokens(
        memory_token_ids=list(memory_token_ids),
        query_token_ids=list(query_tokens.token_ids),
        prompt_token_ids=list(memory_token_ids) + list(query_tokens.token_ids),
        stripped_query_bos=query_tokens.stripped_query_bos,
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
