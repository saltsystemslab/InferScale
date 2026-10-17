"""Retrieved facts and the accounting attached to a composed memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class RetrievedFact:
    """One Mem0 fact as read back from a retrieval hit, with its source turn."""

    memory_id: str
    text: str
    text_hash: str
    created_at: str
    source_session_index: int
    source_session_id: str
    source_turn_index: int
    source_turn_id: str


@dataclass(slots=True, frozen=True)
class MemoryFactPlan:
    """Which facts a memory holds, how they were encoded, and its token accounting."""

    context_window: int
    retrieved_fact_ids: tuple[str, ...]
    retrieved_fact_text_hashes: tuple[str, ...]
    injected_fact_ids: tuple[str, ...]
    context_turn_ids: tuple[str, ...]
    context_encoding_tokens_total: int
    context_encoding_tokens_max: int
    context_encoding_truncated_tokens: int
    scaffold_tokens: int
    fact_tokens: int
    context_text_tokens: int
    memory_tokens: int
    memory_token_budget: int


@dataclass(slots=True)
class ComposedSampleMemory:
    """The KV memory composed for one question from a sample's precomputed chunks."""

    kv_by_layer: dict[str, Any]
    token_ids: list[int]
    num_tokens: int
    compose_time_ms: float
    fact_plan: MemoryFactPlan
    loaded_memory_tokens: int
    recomputed_memory_tail_tokens: int
    fact_tokens_end: int
