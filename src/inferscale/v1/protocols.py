"""Structural interfaces the facade composes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from .kv.memory_store import UserMemory
from .types import SearchHit, SearchMetrics


class Embedder(Protocol):
    """Embeds stored texts and queries; benchmarks inject a cache-backed one."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float] | np.ndarray: ...


class VectorIndex(Protocol):
    """Vector index keyed by chunk id."""

    @property
    def vector_count(self) -> int: ...

    @property
    def dim(self) -> int | None: ...

    def add_many(
        self,
        vectors: Iterable[np.ndarray | list[float]],
        payloads: Iterable[dict[str, Any]],
        ids: Iterable[str] | None = None,
    ) -> list[str]: ...

    def finalize(self) -> None: ...

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchHit], SearchMetrics]: ...

    def get(self, item_id: str) -> SearchHit | None: ...

    def close(self) -> None: ...


class ChunkStore(Protocol):
    """KV payload storage with mandatory CUDA chunk-ID to row lookup.

    The corpus methods resolve IDs through a GPU map for both GPU and pinned
    host payloads.
    The single-user methods also serve the composed-request registry, whose
    bookkeeping is separate from the text-to-KV corpus map.
    """

    num_staging_slots: int

    def add_user_memory(
        self,
        user_id: str,
        kv_by_layer: Mapping[str, Any],
        num_tokens: int,
        token_ids: list[int] | None = None,
    ) -> None: ...

    def get_user_memory(self, user_id: str) -> UserMemory | None: ...

    def peek_user_memory(self, user_id: str) -> UserMemory | None: ...

    def finalize_chunk_lookup(self) -> None: ...

    def get_chunk_memories(
        self, chunk_ids: Sequence[str]
    ) -> list[tuple[str, UserMemory]]: ...

    def build_device_row_map(self, stable_id_items: Iterable[tuple[int, str]]) -> Any: ...

    def get_device_chunk_memories(
        self, stable_ids: Any, id_to_row: Any, *, reverse: bool = False
    ) -> list[tuple[str, UserMemory]]: ...

    def remove_user_memory(self, user_id: str) -> bool: ...

    def get_all_user_ids(self) -> list[str]: ...

    def get_stats(self) -> dict[str, Any]: ...

    def prefetch_user_to_gpu(self, user_id: str) -> bool: ...

    def release_staging(self, user_id: str) -> None: ...

    def get_bench_summary(self) -> dict[str, float | int]: ...

    def transfer_totals(self) -> dict[str, float]: ...

    def transfer_count(self) -> int: ...

    def last_transfer_record(self) -> Any: ...

    def reset_bench_metrics(self) -> None: ...
