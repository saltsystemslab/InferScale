from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from ..data import ConversationSample, Turn, format_turn_for_memory
from ..vector_types import SearchHit
from .tokenization import encode_text_no_special
from .types import (
    EncodedChunk,
    FactContextEncodingPlan,
    MemoryFact,
    MemoryFactPlan,
)


def memory_fact_from_hit(hit: SearchHit) -> MemoryFact:
    memory_id = str(hit.id).strip()
    if not memory_id:
        raise RuntimeError("Mem0 returned a fact with an empty memory id.")

    raw_text = hit.payload.get("memory")
    text = str(raw_text).strip() if raw_text is not None else ""
    if not text:
        raise RuntimeError(f"Mem0 fact {memory_id} has no canonical payload['memory'] text.")

    raw_session_index = _payload_value(
        hit.payload,
        "source_session_index",
        metadata_keys=("source_session_index", "session_index"),
    )
    try:
        source_session_index = int(raw_session_index)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Mem0 fact {memory_id} has no valid source_session_index."
        ) from exc
    if source_session_index < 0:
        raise RuntimeError(
            f"Mem0 fact {memory_id} has negative source_session_index={source_session_index}."
        )

    source_session_id = str(
        _payload_value(
            hit.payload,
            "source_session_id",
            metadata_keys=("source_session_id", "session_id"),
        )
        or f"session_{source_session_index}"
    ).strip()
    source_turn_id = str(
        _payload_value(
            hit.payload,
            "source_turn_id",
            metadata_keys=("source_turn_id", "turn_id"),
        )
        or ""
    ).strip()
    raw_turn_index = _payload_value(
        hit.payload,
        "source_turn_index",
        metadata_keys=("source_turn_index", "turn_index"),
    )
    if raw_turn_index is None and source_turn_id:
        raw_turn_index = source_turn_id.rsplit(":", 1)[-1]
    try:
        source_turn_index = int(raw_turn_index)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Mem0 fact {memory_id} has no valid source_turn_index."
        ) from exc
    if source_turn_index < 0:
        raise RuntimeError(
            f"Mem0 fact {memory_id} has negative source_turn_index={source_turn_index}."
        )
    created_at = str(hit.payload.get("created_at") or "").strip()
    return MemoryFact(
        memory_id=memory_id,
        text=text,
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        created_at=created_at,
        source_session_index=source_session_index,
        source_session_id=source_session_id,
        source_turn_index=source_turn_index,
        source_turn_id=source_turn_id,
    )


def unique_memory_facts(hits: Sequence[SearchHit]) -> list[MemoryFact]:
    facts: list[MemoryFact] = []
    by_id: dict[str, MemoryFact] = {}
    for hit in hits:
        fact = memory_fact_from_hit(hit)
        existing = by_id.get(fact.memory_id)
        if existing is None:
            by_id[fact.memory_id] = fact
            facts.append(fact)
            continue
        if existing != fact:
            raise RuntimeError(
                f"Conflicting Mem0 fact payloads share memory id {fact.memory_id}."
            )
    return facts


def reverse_ranked_memory_facts(hits: Sequence[SearchHit]) -> list[MemoryFact]:
    """Return unique retrieved facts with the best-ranked fact last."""
    return list(reversed(unique_memory_facts(hits)))


def previous_turn_context_turns(
    sample: ConversationSample,
    source_turn_id: str,
    context_window: int,
) -> list[Turn]:
    """Return the turns immediately preceding the given source turn.

    Selection is strictly before the source turn, crosses session boundaries,
    and returns turns in chronological order.
    """
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if context_window == 0:
        return []

    try:
        target_index = next(
            index
            for index, candidate in enumerate(sample.turns)
            if candidate.id == source_turn_id
        )
    except StopIteration as exc:
        raise RuntimeError(
            f"Source turn {source_turn_id} is not present in sample {sample.sample_id}."
        ) from exc

    first_context_index = max(0, target_index - context_window)
    return sample.turns[first_context_index:target_index]


def format_memory_turn(turn: Turn) -> str:
    return format_turn_for_memory(turn).strip() + "\n"


def context_turn_token_ids(
    tokenizer: Any,
    turn: Turn,
    turn_token_ids: MutableMapping[str, list[int]] | None = None,
) -> list[int]:
    if turn_token_ids is not None and turn.id in turn_token_ids:
        return list(turn_token_ids[turn.id])
    tokens = encode_text_no_special(tokenizer, format_memory_turn(turn))
    if turn_token_ids is not None:
        turn_token_ids[turn.id] = list(tokens)
    return tokens


def build_fact_context_encoding_plan(
    target: MemoryFact,
    sample: ConversationSample,
    *,
    context_window: int,
    max_input_tokens: int,
    fact_token_ids: Mapping[str, list[int]],
    sample_facts: Sequence[MemoryFact],
) -> FactContextEncodingPlan:
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

    # fact-encoding-prefix-discard-v1: the prefix is the catalog facts
    # extracted from the window turns, not the turns' raw text. Facts from the
    # target's own turn stay excluded, matching the strictly-before turn window.
    context_turns = previous_turn_context_turns(
        sample,
        target.source_turn_id,
        context_window,
    )
    window_turn_ids = {turn.id for turn in context_turns}
    context_facts = [fact for fact in sample_facts if fact.source_turn_id in window_turn_ids]
    context_token_ids: list[int] = []
    for context_fact in context_facts:
        tokens = fact_token_ids.get(context_fact.memory_id)
        if not tokens:
            raise RuntimeError(
                f"Mem0 context fact tokenized to zero tokens: {context_fact.memory_id}"
            )
        context_token_ids.extend(tokens)

    raw_context_tokens = len(context_token_ids)
    overflow = raw_context_tokens + len(target_token_ids) - max_input_tokens
    context_truncated_tokens = max(0, overflow)
    if context_truncated_tokens:
        context_token_ids = context_token_ids[context_truncated_tokens:]

    input_token_ids = context_token_ids + target_token_ids
    slice_start = len(context_token_ids)
    return FactContextEncodingPlan(
        memory_id=target.memory_id,
        target_token_ids=target_token_ids,
        context_token_ids=context_token_ids,
        input_token_ids=input_token_ids,
        slice_start=slice_start,
        slice_end=len(input_token_ids),
        context_turn_ids=tuple(
            _unique_in_order([fact.source_turn_id for fact in context_facts])
        ),
        raw_context_tokens=raw_context_tokens,
        context_truncated_tokens=context_truncated_tokens,
    )


def build_memory_fact_plan(
    selected_facts: Sequence[MemoryFact],
    *,
    retrieved_facts: Sequence[MemoryFact] | None = None,
    context_window: int,
    memory_token_budget: int,
    scaffold_token_count: int,
    fact_token_ids: Mapping[str, list[int]],
    encoded_chunks: Mapping[str, EncodedChunk] | None = None,
    context_turn_ids: Sequence[str] = (),
    context_text_tokens: int = 0,
) -> MemoryFactPlan:
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if memory_token_budget < 0:
        raise ValueError("memory_token_budget must be >= 0.")
    if scaffold_token_count < 0:
        raise ValueError("scaffold_token_count must be >= 0.")
    retrieved = list(selected_facts if retrieved_facts is None else retrieved_facts)
    retrieved_ids = [fact.memory_id for fact in retrieved]
    selected_ids = [fact.memory_id for fact in selected_facts]
    if len(retrieved_ids) != len(selected_ids) or set(retrieved_ids) != set(selected_ids):
        raise RuntimeError(
            "Retrieved and injected Mem0 facts do not contain the same ids."
        )
    missing_tokens = [
        fact.memory_id
        for fact in selected_facts
        if not fact_token_ids.get(fact.memory_id)
    ]
    if missing_tokens:
        raise RuntimeError(
            "Fact tokens are unavailable for: " + ", ".join(missing_tokens[:5])
        )

    fact_tokens = sum(len(fact_token_ids[fact.memory_id]) for fact in selected_facts)
    memory_tokens = scaffold_token_count + context_text_tokens + fact_tokens
    if memory_tokens > memory_token_budget:
        raise RuntimeError(
            "Memory scaffold, context turns, and retrieved Mem0 facts cannot fit within "
            f"the memory token budget: required={memory_tokens} budget={memory_token_budget} "
            f"scaffold={scaffold_token_count} context={context_text_tokens} facts={fact_tokens}."
        )

    plan_context_turn_ids: list[str] = list(context_turn_ids)
    context_prefix_counts: list[int] = []
    context_truncated_tokens = 0
    if encoded_chunks is not None:
        missing_chunks = [
            fact.memory_id
            for fact in selected_facts
            if fact.memory_id not in encoded_chunks
        ]
        if missing_chunks:
            raise RuntimeError(
                "Retrieved Mem0 facts were not pre-encoded: "
                + ", ".join(missing_chunks[:5])
            )
        for fact in selected_facts:
            chunk = encoded_chunks[fact.memory_id]
            plan_context_turn_ids.extend(chunk.context_turn_ids)
            context_prefix_counts.append(chunk.context_prefix_tokens)
            context_truncated_tokens += chunk.context_prefix_truncated_tokens

    return MemoryFactPlan(
        context_window=context_window,
        retrieved_fact_ids=tuple(retrieved_ids),
        retrieved_fact_text_hashes=tuple(fact.text_hash for fact in retrieved),
        injected_fact_ids=tuple(selected_ids),
        context_turn_ids=tuple(_unique_in_order(plan_context_turn_ids)),
        context_encoding_tokens_total=sum(context_prefix_counts),
        context_encoding_tokens_max=max(context_prefix_counts, default=0),
        context_encoding_truncated_tokens=context_truncated_tokens,
        scaffold_tokens=scaffold_token_count,
        fact_tokens=fact_tokens,
        context_text_tokens=context_text_tokens,
        memory_tokens=memory_tokens,
        memory_token_budget=memory_token_budget,
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


def _payload_value(
    payload: Mapping[str, Any],
    top_level_key: str,
    *,
    metadata_keys: Sequence[str],
) -> Any:
    value = payload.get(top_level_key)
    if value is not None:
        return value
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    for key in metadata_keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def _unique_in_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
