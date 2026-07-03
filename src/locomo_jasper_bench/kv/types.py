from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EncodedChunk:
    token_ids: list[int]
    kv_by_layer: dict[str, Any]
    context_window: int = 0
    context_prefix_tokens: int = 0
    raw_context_prefix_tokens: int = 0
    context_prefix_truncated_tokens: int = 0


@dataclass(slots=True, frozen=True)
class ContextEncodingPlan:
    chunk_id: str
    target_text: str
    target_token_ids: list[int]
    context_token_ids: list[int]
    input_token_ids: list[int]
    slice_start: int
    slice_end: int
    raw_context_prefix_tokens: int
    context_prefix_truncated_tokens: int


@dataclass(slots=True)
class ComposedMemory:
    kv_by_layer: dict[str, Any]
    token_ids: list[int]
    num_tokens: int
    compose_time_ms: float
    retrieval_session_ids: list[str]
    selected_session_ids: list[str]
    memory_order: str
    context_window: int
    context_prefix_tokens_total: int
    context_prefix_tokens_max: int
    context_prefix_truncated_tokens: int
