"""The chunk corpus: KV tensors in a chunk store plus per-chunk metadata."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from ..types import KVChunk, ScaffoldChunks
from .chunk_store import (
    chunk_nbytes,
    close_chunk_store,
    fetch_chunks,
    finalize_chunk_store,
    register_chunks,
    release_chunks,
)

if TYPE_CHECKING:
    from ..protocols import ChunkStore


class KVCorpus:
    """Chunk KV in a store with a GPU ID map, plus local planning metadata.

    ``add`` moves a chunk's tensors into the store and keeps a metadata-only
    copy (token ids and context provenance) for planning and inspection. The
    scaffold chunks (header, empty context, footer) stay as local GPU tensors
    and never enter the store. The store resolves text IDs to KV rows on the
    GPU whether the payloads live on the GPU or in pinned host memory.
    """

    def __init__(self, store: ChunkStore) -> None:
        self._store = store
        self._meta: dict[str, KVChunk] = {}
        self._registered_bytes = 0
        self._registered_layers = 0
        self.scaffold: ScaffoldChunks | None = None

    @property
    def store(self) -> ChunkStore:
        return self._store

    def set_scaffold(self, scaffold: ScaffoldChunks) -> None:
        self.scaffold = scaffold

    def add(self, chunk: KVChunk) -> None:
        if chunk.chunk_id in self._meta:
            raise ValueError(f"Chunk {chunk.chunk_id!r} is already in the corpus.")
        # Registration empties chunk.kv_by_layer, so measure first.
        self._registered_bytes += chunk_nbytes(chunk)
        self._registered_layers = max(self._registered_layers, len(chunk.kv_by_layer))
        self._meta.update(register_chunks(self._store, {chunk.chunk_id: chunk}))

    def ids(self) -> list[str]:
        return list(self._meta)

    def __len__(self) -> int:
        return len(self._meta)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._meta

    def get(self, chunk_id: str) -> KVChunk | None:
        """Metadata-only chunk (empty kv_by_layer)."""
        return self._meta.get(chunk_id)

    def fetch(self, chunk_ids: Sequence[str]) -> list[KVChunk]:
        """Resolve IDs through the store's GPU map and load their KV payloads."""
        return fetch_chunks(self._store, self._meta, list(chunk_ids))

    def release(self, chunk_ids: Iterable[str]) -> None:
        release_chunks(self._store, chunk_ids)

    def finalize(self) -> None:
        """Finalize the payload layout and build the GPU lookup."""
        finalize_chunk_store(self._store)

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "chunk_count": len(self._meta),
            "chunk_bytes": self._registered_bytes,
            "layers": self._registered_layers,
        }
        for key, value in self._store.get_stats().items():
            stats[f"store_{key}"] = value
        return stats

    def close(self) -> None:
        close_chunk_store(self._store)
        self._meta.clear()
        self.scaffold = None
        self._registered_bytes = 0
        self._registered_layers = 0
