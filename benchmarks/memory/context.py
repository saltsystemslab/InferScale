"""Conversation context around a fact: the preceding turns, encoding plans, and memory plans.

A fact is encoded with the ``context_window`` turns that precede its source
turn as an encoding-only prefix. KV injection keeps only the fact's own
tokens from that encode; prompt injection renders the context turns as text.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from inferscale.v1.kv.plan import build_encoding_plan
from inferscale.v1.kv.tokenization import encode_text_no_special
from inferscale.v1.types import EncodingPlan, KVChunk

from .data import ConversationSample, Turn, format_turn_for_memory
from .types import MemoryFactPlan, RetrievedFact


def previous_turn_context_turns(
    sample: ConversationSample,
    source_turn_id: str,
    context_window: int,
) -> list[Turn]:
    """The turns immediately before the source turn, across sessions, in chronological order."""
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if context_window == 0:
        return []
    target_index = next(
        (index for index, turn in enumerate(sample.turns) if turn.id == source_turn_id), None
    )
    if target_index is None:
        raise RuntimeError(
            f"Source turn {source_turn_id} is not present in sample {sample.sample_id}."
        )
    return sample.turns[max(0, target_index - context_window) : target_index]


def format_memory_turn(turn: Turn) -> str:
    return format_turn_for_memory(turn).strip() + "\n"


def context_turn_token_ids(
    tokenizer: Any,
    turn: Turn,
    turn_token_ids: MutableMapping[str, list[int]] | None = None,
) -> list[int]:
    """Tokens of one context turn, memoized in ``turn_token_ids`` when given."""
    if turn_token_ids is not None and turn.id in turn_token_ids:
        return list(turn_token_ids[turn.id])
    tokens = encode_text_no_special(tokenizer, format_memory_turn(turn))
    if turn_token_ids is not None:
        turn_token_ids[turn.id] = list(tokens)
    return tokens


def build_fact_context_encoding_plan(
    target: RetrievedFact,
    sample: ConversationSample,
    *,
    tokenizer: Any,
    context_window: int,
    max_input_tokens: int,
    fact_token_ids: Mapping[str, list[int]],
    turn_token_ids: MutableMapping[str, list[int]] | None = None,
) -> EncodingPlan:
    """Plan one fact's encode with its preceding conversation turns as the prefix."""
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be >= 1.")
    target_token_ids = list(fact_token_ids.get(target.memory_id, []))
    if not target_token_ids:
        raise RuntimeError(f"Mem0 fact tokenized to zero tokens: {target.memory_id}")
    if len(target_token_ids) > max_input_tokens:
        raise RuntimeError(
            f"Mem0 fact {target.memory_id} has {len(target_token_ids)} tokens, "
            f"exceeding kv_max_position={max_input_tokens} without context."
        )
    context_turns = previous_turn_context_turns(sample, target.source_turn_id, context_window)
    context_token_ids: list[int] = []
    for turn in context_turns:
        context_token_ids.extend(context_turn_token_ids(tokenizer, turn, turn_token_ids))
    return build_encoding_plan(
        target.memory_id,
        target_token_ids,
        context_token_ids=context_token_ids,
        context_ids=tuple(turn.id for turn in context_turns),
        max_input_tokens=max_input_tokens,
    )


def plan_kv_memory(
    selected_facts: Sequence[RetrievedFact],
    *,
    retrieved_facts: Sequence[RetrievedFact] | None = None,
    context_window: int,
    memory_token_budget: int,
    scaffold_token_count: int,
    fact_token_ids: Mapping[str, list[int]],
    encoded_chunks: Mapping[str, KVChunk],
) -> MemoryFactPlan:
    """The plan for a memory composed from pre-encoded chunks.

    Context turns never enter the memory as tokens; the plan records which
    turns conditioned each chunk's encode and how many prefix tokens that took.
    """
    retrieved = _checked_facts(selected_facts, retrieved_facts, fact_token_ids)
    missing = [fact.memory_id for fact in selected_facts if fact.memory_id not in encoded_chunks]
    if missing:
        raise RuntimeError("Retrieved Mem0 facts were not pre-encoded: " + ", ".join(missing[:5]))
    chunks = [encoded_chunks[fact.memory_id] for fact in selected_facts]
    prefix_counts = [chunk.context_prefix_tokens for chunk in chunks]
    return _plan(
        selected_facts,
        retrieved,
        context_window=context_window,
        memory_token_budget=memory_token_budget,
        scaffold_token_count=scaffold_token_count,
        fact_token_ids=fact_token_ids,
        context_turn_ids=[turn_id for chunk in chunks for turn_id in chunk.context_ids],
        context_encoding_tokens_total=sum(prefix_counts),
        context_encoding_tokens_max=max(prefix_counts, default=0),
        context_encoding_truncated_tokens=sum(
            chunk.context_prefix_truncated_tokens for chunk in chunks
        ),
        context_text_tokens=0,
    )


def plan_prompt_memory(
    selected_facts: Sequence[RetrievedFact],
    *,
    retrieved_facts: Sequence[RetrievedFact] | None = None,
    context_window: int,
    memory_token_budget: int,
    scaffold_token_count: int,
    fact_token_ids: Mapping[str, list[int]],
    context_turn_ids: Sequence[str] = (),
    context_text_tokens: int = 0,
) -> MemoryFactPlan:
    """The plan for a memory rendered as prompt text, context turns included as tokens."""
    retrieved = _checked_facts(selected_facts, retrieved_facts, fact_token_ids)
    return _plan(
        selected_facts,
        retrieved,
        context_window=context_window,
        memory_token_budget=memory_token_budget,
        scaffold_token_count=scaffold_token_count,
        fact_token_ids=fact_token_ids,
        context_turn_ids=list(context_turn_ids),
        context_encoding_tokens_total=0,
        context_encoding_tokens_max=0,
        context_encoding_truncated_tokens=0,
        context_text_tokens=context_text_tokens,
    )


def memory_context_metrics(plan: MemoryFactPlan) -> dict[str, Any]:
    return {
        "memory_context_window": plan.context_window,
        "memory_retrieved_fact_ids": list(plan.retrieved_fact_ids),
        "memory_retrieved_fact_text_hashes": list(plan.retrieved_fact_text_hashes),
        "memory_injected_fact_ids": list(plan.injected_fact_ids),
        "memory_context_turn_ids": list(plan.context_turn_ids),
        "memory_context_turn_count": len(plan.context_turn_ids),
        "memory_context_encoding_tokens_total": plan.context_encoding_tokens_total,
        "memory_context_encoding_tokens_max": plan.context_encoding_tokens_max,
        "memory_context_encoding_truncated_tokens": plan.context_encoding_truncated_tokens,
        "memory_context_text_tokens": plan.context_text_tokens,
        "memory_token_budget": plan.memory_token_budget,
    }


def _checked_facts(
    selected_facts: Sequence[RetrievedFact],
    retrieved_facts: Sequence[RetrievedFact] | None,
    fact_token_ids: Mapping[str, list[int]],
) -> list[RetrievedFact]:
    """The retrieved facts, after checking they are the injected facts and all tokenized."""
    retrieved = list(selected_facts if retrieved_facts is None else retrieved_facts)
    retrieved_ids = [fact.memory_id for fact in retrieved]
    selected_ids = [fact.memory_id for fact in selected_facts]
    if len(retrieved_ids) != len(selected_ids) or set(retrieved_ids) != set(selected_ids):
        raise RuntimeError("Retrieved and injected Mem0 facts do not contain the same ids.")
    missing = [fact.memory_id for fact in selected_facts if not fact_token_ids.get(fact.memory_id)]
    if missing:
        raise RuntimeError("Fact tokens are unavailable for: " + ", ".join(missing[:5]))
    return retrieved


def _plan(
    selected_facts: Sequence[RetrievedFact],
    retrieved_facts: Sequence[RetrievedFact],
    *,
    context_window: int,
    memory_token_budget: int,
    scaffold_token_count: int,
    fact_token_ids: Mapping[str, list[int]],
    context_turn_ids: list[str],
    context_encoding_tokens_total: int,
    context_encoding_tokens_max: int,
    context_encoding_truncated_tokens: int,
    context_text_tokens: int,
) -> MemoryFactPlan:
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if memory_token_budget < 0:
        raise ValueError("memory_token_budget must be >= 0.")
    if scaffold_token_count < 0:
        raise ValueError("scaffold_token_count must be >= 0.")
    fact_tokens = sum(len(fact_token_ids[fact.memory_id]) for fact in selected_facts)
    memory_tokens = scaffold_token_count + context_text_tokens + fact_tokens
    if memory_tokens > memory_token_budget:
        raise RuntimeError(
            "Memory scaffold, context turns, and retrieved Mem0 facts cannot fit within "
            f"the memory token budget: required={memory_tokens} budget={memory_token_budget} "
            f"scaffold={scaffold_token_count} context={context_text_tokens} facts={fact_tokens}."
        )
    return MemoryFactPlan(
        context_window=context_window,
        retrieved_fact_ids=tuple(fact.memory_id for fact in retrieved_facts),
        retrieved_fact_text_hashes=tuple(fact.text_hash for fact in retrieved_facts),
        injected_fact_ids=tuple(fact.memory_id for fact in selected_facts),
        context_turn_ids=tuple(dict.fromkeys(turn_id for turn_id in context_turn_ids if turn_id)),
        context_encoding_tokens_total=context_encoding_tokens_total,
        context_encoding_tokens_max=context_encoding_tokens_max,
        context_encoding_truncated_tokens=context_encoding_truncated_tokens,
        scaffold_tokens=scaffold_token_count,
        fact_tokens=fact_tokens,
        context_text_tokens=context_text_tokens,
        memory_tokens=memory_tokens,
        memory_token_budget=memory_token_budget,
    )
