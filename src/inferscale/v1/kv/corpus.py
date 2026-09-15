"""The chunk corpus: KV tensors in a chunk store plus per-chunk metadata."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from ..types import KVChunk, ScaffoldChunks
from .chunk_store import (
    build_device_chunk_row_map,
    chunk_nbytes,
    close_chunk_store,
    fetch_chunks,
    fetch_device_chunk,
    fetch_device_chunks,
    finalize_chunk_store,
    register_chunks,
    release_chunks,
)

if TYPE_CHECKING:
    from ..protocols import ChunkStore


class KVCorpus:
    """Chunk KV with a GPU ID map and local planning metadata.

    ``add`` moves a chunk's tensors into the store and keeps a metadata-only
    copy (token ids and context provenance) for planning and inspection. The
    scaffold chunks (header, empty context, footer) stay as local GPU tensors
    and never enter the store, exactly like the corpus stores they wrap.
    The store resolves text IDs to numeric KV rows on GPU regardless of
    whether the corpus payloads are on GPU or in pinned host memory.
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
        """Build the mandatory GPU lookup and finalize the payload layout."""
        finalize_chunk_store(self._store)

    def build_device_row_map(self, stable_id_items: Iterable[tuple[int, str]]) -> Any:
        return build_device_chunk_row_map(self._store, stable_id_items)

    def fetch_device_chunk(self, stable_ids: Any, id_to_row: Any) -> KVChunk:
        return fetch_device_chunk(self._store, stable_ids, id_to_row)

    def fetch_device_chunks(
        self, stable_ids: Any, id_to_row: Any, *, reverse: bool = True
    ) -> list[KVChunk]:
        """Resolve Jasper IDs on GPU for either corpus payload backend."""
        return fetch_device_chunks(
            self._store, self._meta, stable_ids, id_to_row, reverse=reverse
        )

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "chunk_count": len(self._meta),
            "chunk_bytes": self._registered_bytes,
            "layers": self._registered_layers,
        }
        for key, value in self._store.get_stats().items():
            stats[f"store_{key}"] = value
        return stats

    def transfer_totals(self) -> dict[str, float]:
        return dict(self._store.transfer_totals())

    def bench_summary(self) -> dict[str, Any]:
        return dict(self._store.get_bench_summary())

    def close(self) -> None:
        close_chunk_store(self._store)
        self._meta.clear()
        self.scaffold = None
        self._registered_bytes = 0
        self._registered_layers = 0
