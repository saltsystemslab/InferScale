"""Structural interfaces the facade composes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from .kv.memory_store import UserMemory
from .types import SearchHit, SearchMetrics


class Embedder(Protocol):
    """Embeds stored texts and queries; callers may wrap one in a cache."""

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


class MemoryStore(Protocol):
    """Registered KV memories as the connector reads them.

    A store without staging reports zero ``num_staging_slots`` and treats
    prefetch and release as no-ops.
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

    def remove_user_memory(self, user_id: str) -> bool: ...

    def get_all_user_ids(self) -> list[str]: ...

    def get_stats(self) -> dict[str, Any]: ...

    def prefetch_user_to_gpu(self, user_id: str) -> bool: ...

    def release_staging(self, user_id: str) -> None: ...


class ChunkStore(MemoryStore, Protocol):
    """A memory store that also serves the corpus through a GPU ID lookup."""

    def finalize_packed(self) -> None: ...

    def finalize_chunk_lookup(self) -> None: ...

    def close_chunk_lookup(self) -> None: ...

    def get_chunk_memories(self, chunk_ids: Sequence[str]) -> list[tuple[str, UserMemory]]: ...

    def build_device_row_map(self, stable_id_items: Iterable[tuple[int, str]]) -> Any: ...

    def get_device_chunk_memories(
        self, stable_ids: Any, id_to_row: Any, *, reverse: bool = False
    ) -> list[tuple[str, UserMemory]]: ...

    def get_bench_summary(self) -> dict[str, float | int]: ...

    def transfer_totals(self) -> dict[str, float]: ...
