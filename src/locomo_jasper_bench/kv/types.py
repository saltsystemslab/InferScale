from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class MemoryFact:
    memory_id: str
    text: str
    text_hash: str
    created_at: str
    source_session_index: int
    source_session_id: str
    source_turn_index: int
    source_turn_id: str


@dataclass(slots=True, frozen=True)
class FactContextEncodingPlan:
    memory_id: str
    target_token_ids: list[int]
    context_token_ids: list[int]
    input_token_ids: list[int]
    slice_start: int
    slice_end: int
    context_turn_ids: tuple[str, ...]
    raw_context_tokens: int
    context_truncated_tokens: int


@dataclass(slots=True)
class EncodedChunk:
    token_ids: list[int]
    kv_by_layer: dict[str, Any]
    context_turn_ids: tuple[str, ...] = ()
    context_prefix_tokens: int = 0
    raw_context_prefix_tokens: int = 0
    context_prefix_truncated_tokens: int = 0


@dataclass(slots=True, frozen=True)
class MemoryFactPlan:
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
class ComposedMemory:
    kv_by_layer: dict[str, Any]
    token_ids: list[int]
    num_tokens: int
    compose_time_ms: float
    fact_plan: MemoryFactPlan
    loaded_memory_tokens: int
    recomputed_memory_tail_tokens: int
    fact_tokens_end: int

    @property
    def selected_fact_ids(self) -> list[str]:
        return list(self.fact_plan.injected_fact_ids)

    @property
    def context_window(self) -> int:
        return self.fact_plan.context_window
