"""Backend-selected store for the chunk corpus.

The chunk store is the home of the corpus: chunks encoded once (or loaded
from a disk cache) and reused across requests. The ``gpu`` backend keeps the
corpus HBM-resident (GPUMemoryStore); ``cpu`` keeps it in pinned host RAM
(CpuPinnedMemoryStore) and stages the selected chunks to the GPU per
composition, which is where a corpus-in-DRAM system pays its PCIe cost.

Composed request memories never enter these stores; they are ephemeral GPU
products handed to the connector through the (always GPU-resident) serving
namespace registry and discarded after injection.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..types import KVChunk
from .memory_store import GPUMemoryStore, tensor_nbytes
from .packed_memory_store import PackedGPUMemoryStore

# Slots beyond one full top-k fetch, covering allocator slack and the next
# request's first prefetches.
_STAGING_HEADROOM = 4


def build_chunk_store(
    backend: str,
    *,
    device: str,
    top_k: int,
    staging_slots: int = 0,
    device_selection: bool = False,
) -> Any:
    """A dedicated store instance for the corpus, never a connector namespace."""
    if backend == "cpu":
        from .pinned_memory_store import CpuPinnedMemoryStore

        # Every chunk of one composition must be staged simultaneously, so
        # the pool must cover a full top-k fetch; the configured slot count
        # can only raise the floor, never lower it.
        return CpuPinnedMemoryStore(
            device=device,
            num_staging_slots=max(int(staging_slots), int(top_k) + _STAGING_HEADROOM),
        )
    if backend == "gpu":
        if device_selection:
            return PackedGPUMemoryStore(device=device)
        return GPUMemoryStore(device=device)
    raise ValueError(f"Unknown chunk store backend: {backend!r}; expected 'gpu' or 'cpu'.")


def register_chunks(
    store: Any,
    chunks: Mapping[str, KVChunk],
) -> dict[str, KVChunk]:
    """Move chunk KV into the store; return the metadata-only chunk map.

    The metadata map keeps token_ids and the context_* fields (everything
    the planning consumers read) with an empty kv_by_layer; the tensors live
    in the store until close_chunk_store.
    """
    meta: dict[str, KVChunk] = {}
    for chunk_id, chunk in chunks.items():
        if chunk_id != chunk.chunk_id:
            raise ValueError(
                f"Chunk map key {chunk_id!r} does not match chunk_id {chunk.chunk_id!r}."
            )
        chunk_meta = _metadata_only(chunk)
        store.add_user_memory(
            user_id=chunk_id,
            kv_by_layer=chunk.kv_by_layer,
            num_tokens=len(chunk.token_ids),
            token_ids=chunk.token_ids,
        )
        # Registration transfers tensor ownership to the store. Dropping the
        # source references matters once a packed store allocates its slabs:
        # otherwise cache payloads would keep the pre-pack tensors alive.
        chunk.kv_by_layer = {}
        meta[chunk_id] = chunk_meta
    return meta


def fetch_chunks(
    store: Any,
    meta: Mapping[str, KVChunk],
    chunk_ids: Sequence[str],
) -> list[KVChunk]:
    """Stage the selected chunks and return compose-ready chunks.

    On the cpu store this issues the async H2D copies; the returned chunks'
    kv views wait per layer on first access. Call release_chunks with the
    same ids once the composition kernels have been synchronized.
    """
    capacity = int(getattr(store, "num_staging_slots", 0) or 0)
    if capacity and len(chunk_ids) > capacity:
        raise RuntimeError(
            f"Composition needs {len(chunk_ids)} staged chunks but the chunk "
            f"store has only {capacity} staging slots; raise the staging slot count."
        )
    fetched: list[KVChunk] = []
    for chunk_id in chunk_ids:
        chunk_meta = meta.get(chunk_id)
        if chunk_meta is None:
            raise RuntimeError(f"Retrieved chunk id {chunk_id} has no pre-encoded KV chunk.")
        memory = store.get_user_memory(chunk_id)
        if memory is None:
            raise RuntimeError(f"Chunk {chunk_id} is missing from the chunk store.")
        fetched.append(
            KVChunk(
                chunk_id=chunk_id,
                token_ids=chunk_meta.token_ids,
                # A plain dict on the gpu store; a staged per-layer view on
                # the cpu store. Compose consumes both identically.
                kv_by_layer=memory.kv_by_layer,
                context_ids=chunk_meta.context_ids,
                context_prefix_tokens=chunk_meta.context_prefix_tokens,
                raw_context_prefix_tokens=chunk_meta.raw_context_prefix_tokens,
                context_prefix_truncated_tokens=chunk_meta.context_prefix_truncated_tokens,
            )
        )
    return fetched


def release_chunks(store: Any, chunk_ids: Iterable[str]) -> None:
    for chunk_id in chunk_ids:
        store.release_staging(chunk_id)


def finalize_chunk_store(store: Any) -> None:
    """Finalize an optional packed layout after all corpus chunks are registered."""
    finalize = getattr(store, "finalize_packed", None)
    if callable(finalize):
        finalize()


def build_device_chunk_row_map(
    store: Any,
    stable_id_items: Iterable[tuple[int, str]],
) -> Any:
    build_map = getattr(store, "build_device_row_map", None)
    if not callable(build_map):
        raise RuntimeError("The chunk store does not support device-side row mapping.")
    return build_map(stable_id_items)


def fetch_device_chunk(
    store: Any,
    stable_ids: Any,
    id_to_row: Any,
) -> KVChunk:
    """Gather reverse-ranked chunk KV and token ids from a packed GPU corpus."""
    select = getattr(store, "select_device_ids", None)
    if not callable(select):
        raise RuntimeError("The chunk store does not support device-side selection.")
    memory = select(stable_ids, id_to_row, reverse=True)
    if not memory.token_ids:
        raise RuntimeError("Jasper device selection returned no chunk tokens.")
    return KVChunk(
        chunk_id="device-selection",
        token_ids=list(memory.token_ids),
        kv_by_layer=memory.kv_by_layer,
    )


def close_chunk_store(store: Any) -> None:
    for chunk_id in list(store.get_all_user_ids()):
        store.remove_user_memory(chunk_id)


def chunk_nbytes(chunk: KVChunk | None) -> int:
    if chunk is None:
        return 0
    return sum(tensor_nbytes(tensor) for tensor in chunk.kv_by_layer.values())


def _metadata_only(chunk: KVChunk) -> KVChunk:
    return KVChunk(
        chunk_id=chunk.chunk_id,
        token_ids=list(chunk.token_ids),
        kv_by_layer={},
        context_ids=chunk.context_ids,
        context_prefix_tokens=chunk.context_prefix_tokens,
        raw_context_prefix_tokens=chunk.raw_context_prefix_tokens,
        context_prefix_truncated_tokens=chunk.context_prefix_truncated_tokens,
    )
