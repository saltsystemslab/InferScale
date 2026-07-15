"""Backend-selected store for the fact-chunk corpus.

The memory stores are the home of the memory corpus: fact chunks
encoded once (or loaded from the disk chunk cache) and reused across
requests. `--kv-store-backend gpu` keeps the corpus HBM-resident
(GPUMemoryStore); `cpu` keeps it in pinned host RAM (CpuPinnedMemoryStore)
and stages the selected chunks to the GPU per composition, which is where
a corpus-in-DRAM system pays its PCIe cost.

Composed request memories never enter these stores - they are ephemeral
GPU products handed to the connector through the (always GPU-resident)
serving namespace registry and discarded after injection.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .gpu_memory_store import GPUMemoryStore
from .types import EncodedChunk

# Slots beyond one full top-k fetch, covering allocator slack and the next
# request's first prefetches.
_STAGING_HEADROOM = 4


def build_chunk_store(
    backend: str,
    *,
    device: str,
    top_k: int,
    staging_slots: int = 0,
) -> Any:
    """A dedicated store instance for the corpus - never a connector namespace."""
    if backend == "cpu":
        from .cpu_memory_store import CpuPinnedMemoryStore

        # Every chunk of one composition must be staged simultaneously, so
        # the pool must cover a full top-k fetch; --kv-staging-slots can only
        # raise the floor, never lower it.
        return CpuPinnedMemoryStore(
            device=device,
            num_staging_slots=max(int(staging_slots), int(top_k) + _STAGING_HEADROOM),
        )
    if backend == "gpu":
        return GPUMemoryStore(device=device)
    raise ValueError(f"Unknown chunk store backend: {backend!r}; expected 'gpu' or 'cpu'.")


def register_fact_chunks(
    store: Any,
    fact_chunks: Mapping[str, EncodedChunk],
) -> dict[str, EncodedChunk]:
    """Move chunk KV into the store; return the metadata-only chunk map.

    The metadata map keeps token_ids and the context_* fields (everything
    the planning/verification consumers read) with an empty kv_by_layer;
    the tensors live in the store until close_chunk_store.
    """
    meta: dict[str, EncodedChunk] = {}
    for fact_id, chunk in fact_chunks.items():
        store.add_user_memory(
            user_id=fact_id,
            kv_by_layer=chunk.kv_by_layer,
            num_tokens=len(chunk.token_ids),
            token_ids=chunk.token_ids,
        )
        meta[fact_id] = _metadata_only(chunk)
    return meta


def fetch_fact_chunks(
    store: Any,
    meta: Mapping[str, EncodedChunk],
    fact_ids: Sequence[str],
) -> list[EncodedChunk]:
    """Stage the selected chunks and return compose-ready EncodedChunks.

    On the cpu store this issues the async H2D copies; the returned chunks'
    kv views wait per layer on first access. Call release_fact_chunks with
    the same ids once the composition kernels have been synchronized.
    """
    capacity = int(getattr(store, "num_staging_slots", 0) or 0)
    if capacity and len(fact_ids) > capacity:
        raise RuntimeError(
            f"Composition needs {len(fact_ids)} staged chunks but the chunk "
            f"store has only {capacity} staging slots; raise --kv-staging-slots."
        )
    fetched: list[EncodedChunk] = []
    for fact_id in fact_ids:
        chunk_meta = meta.get(fact_id)
        if chunk_meta is None:
            raise RuntimeError(f"Retrieved fact_id={fact_id} has no pre-encoded KV chunk.")
        memory = store.get_user_memory(fact_id)
        if memory is None:
            raise RuntimeError(f"Fact chunk {fact_id} is missing from the chunk store.")
        fetched.append(
            EncodedChunk(
                token_ids=chunk_meta.token_ids,
                # A plain dict on the gpu store; a staged per-layer view on
                # the cpu store. Compose consumes both identically.
                kv_by_layer=memory.kv_by_layer,
                context_turn_ids=chunk_meta.context_turn_ids,
                context_prefix_tokens=chunk_meta.context_prefix_tokens,
                raw_context_prefix_tokens=chunk_meta.raw_context_prefix_tokens,
                context_prefix_truncated_tokens=chunk_meta.context_prefix_truncated_tokens,
            )
        )
    return fetched


def release_fact_chunks(store: Any, fact_ids: Iterable[str]) -> None:
    for fact_id in fact_ids:
        store.release_staging(fact_id)


def close_chunk_store(store: Any) -> None:
    for fact_id in list(store.get_all_user_ids()):
        store.remove_user_memory(fact_id)


def _metadata_only(chunk: EncodedChunk) -> EncodedChunk:
    return EncodedChunk(
        token_ids=list(chunk.token_ids),
        kv_by_layer={},
        context_turn_ids=chunk.context_turn_ids,
        context_prefix_tokens=chunk.context_prefix_tokens,
        raw_context_prefix_tokens=chunk.raw_context_prefix_tokens,
        context_prefix_truncated_tokens=chunk.context_prefix_truncated_tokens,
    )
