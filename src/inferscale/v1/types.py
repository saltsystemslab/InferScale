from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

VECTOR_DISTANCE = "ip"


@dataclass(slots=True, frozen=True)
class Chunk:
    id: str
    text: str
    token_ids: Sequence[int] | None = None
    payload: Mapping[str, Any] | None = field(default=None, kw_only=True)


@dataclass(slots=True, frozen=True)
class EncodingPlan:
    chunk_id: str
    target_token_ids: list[int]
    context_token_ids: list[int]
    input_token_ids: list[int]
    slice_start: int
    slice_end: int
    context_ids: tuple[str, ...]
    raw_context_tokens: int
    context_truncated_tokens: int


@dataclass(slots=True)
class KVChunk:
    chunk_id: str
    token_ids: list[int]
    kv_by_layer: dict[str, Any]
    context_ids: tuple[str, ...] = ()
    context_prefix_tokens: int = 0
    raw_context_prefix_tokens: int = 0
    context_prefix_truncated_tokens: int = 0


@dataclass(slots=True, frozen=True)
class ScaffoldTokens:
    header_token_ids: list[int]
    empty_token_ids: list[int]
    footer_token_ids: list[int]


@dataclass(slots=True)
class ScaffoldChunks:
    header: KVChunk
    empty: KVChunk
    footer: KVChunk


@dataclass(slots=True, frozen=True)
class QueryTokens:
    token_ids: list[int]
    stripped_query_bos: bool


@dataclass(slots=True, frozen=True)
class PromptTokens:
    memory_token_ids: list[int]
    query_token_ids: list[int]
    prompt_token_ids: list[int]
    stripped_query_bos: bool


@dataclass(slots=True)
class ComposedMemory:
    kv_by_layer: dict[str, Any]
    token_ids: list[int]
    num_tokens: int
    compose_time_ms: float
    loaded_memory_tokens: int
    recomputed_tail_tokens: int


@dataclass(slots=True)
class SearchMetrics:
    search_time_ms: float
    vector_backend: str | None = None
    jasper_effective_beam_width: int | None = None


@dataclass(slots=True, frozen=True)
class DeviceSearchResult:
    stable_ids: Any
    distances: Any
    metrics: SearchMetrics


@dataclass(slots=True)
class RetrievalMetrics:
    embedding_time_ms: float
    search_time_ms: float
    total_time_ms: float
    vector_backend: str | None = None
    jasper_effective_beam_width: int | None = None


@dataclass(slots=True)
class SearchHit:
    id: str
    payload: dict[str, Any]
    score: float
    distance: float
    rank: int


@dataclass(slots=True)
class QueryResult:
    text: str
    ttft_ms: float
    hits: list[SearchHit]
    retrieval: RetrievalMetrics | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PrecomputeStats:
    chunk_count: int
    token_count: int
    kv_bytes: int
    encode_time_ms: float
    embed_time_ms: float
    store: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ChunkInfo:
    id: str
    text: str
    token_ids: list[int]
    context_ids: tuple[str, ...]
    context_prefix_tokens: int
    payload: dict[str, Any]
