"""Size and residency statistics of a sample's precomputed KV chunks."""

from __future__ import annotations

import sys
from typing import Any

from inferscale.v1.kv.chunk_store import chunk_nbytes
from inferscale.v1.types import KVChunk
from inferscale.v1.units import bytes_to_mb

from .composer import SampleComposer


def precompute_stats(composer: SampleComposer) -> dict[str, Any]:
    """The setup metrics a precomputed sample reports, keyed as the result schema names them."""
    scaffold_chunks = composer.encoded_scaffold_chunks()
    fact_chunks = list(composer.chunks.values())
    chunks = [*scaffold_chunks, *fact_chunks]
    layer_count = max((len(chunk.kv_by_layer) for chunk in chunks), default=0)
    devices = {str(tensor.device) for chunk in chunks for tensor in chunk.kv_by_layer.values()}
    store = composer.chunk_store
    if composer.chunks_in_store:
        # The fact-chunk KV lives in the store and the local chunks are
        # metadata-only: report the sizes captured at move time and the
        # store's residency.
        fact_chunk_bytes = composer.stored_chunk_bytes
        layer_count = max(layer_count, composer.stored_chunk_layers)
        residency = "cpu" if store is not None and store.num_staging_slots else "gpu"
        devices.add("cpu(pinned)" if residency == "cpu" else str(composer.device))
    else:
        fact_chunk_bytes = sum(chunk_nbytes(chunk) for chunk in fact_chunks)
        residency = "gpu"
    prefix_bytes = sum(chunk_nbytes(chunk) for chunk in scaffold_chunks)
    total_bytes = prefix_bytes + fact_chunk_bytes
    metadata_bytes = _chunk_metadata_cpu_bytes(composer.chunks)
    store_stats = store.get_stats() if store is not None else {}
    chunk_map_bytes = int(store_stats.get("chunk_map_bytes", 0))
    stats = {
        "kv_chunk_cache_residency": residency,
        "kv_precomputed_chunks": len(fact_chunks),
        "kv_precomputed_chunks_with_prefix": len(chunks),
        "kv_precomputed_tokens": sum(len(chunk.token_ids) for chunk in chunks),
        "kv_precomputed_layers": layer_count,
        "kv_precomputed_gpu_mb": total_bytes / (1024 * 1024),
        "kv_precomputed_devices": ",".join(sorted(devices)),
        "llama_kv_chunk_count": len(fact_chunks),
        "llama_kv_chunk_metadata_cpu_bytes": metadata_bytes,
        "llama_kv_chunk_metadata_cpu_mb": bytes_to_mb(metadata_bytes),
        "llama_kv_chunk_map_gpu_bytes": chunk_map_bytes,
        "llama_kv_chunk_map_gpu_mb": bytes_to_mb(chunk_map_bytes),
        "llama_kv_chunk_tensor_gpu_bytes": fact_chunk_bytes,
        "llama_kv_chunk_tensor_gpu_mb": bytes_to_mb(fact_chunk_bytes),
        "llama_kv_prefix_tensor_gpu_bytes": prefix_bytes,
        "llama_kv_prefix_tensor_gpu_mb": bytes_to_mb(prefix_bytes),
        "llama_kv_total_tensor_gpu_bytes": total_bytes,
        "llama_kv_total_tensor_gpu_mb": bytes_to_mb(total_bytes),
    }
    for key, value in store_stats.items():
        stats[f"kv_chunk_store_{key}"] = value
    return stats


def _chunk_metadata_cpu_bytes(chunks_by_id: dict[str, KVChunk]) -> int:
    """Host bytes of the chunk map and its metadata-only chunks, as Python objects."""
    total = sys.getsizeof(chunks_by_id)
    for chunk_id, chunk in chunks_by_id.items():
        total += sys.getsizeof(chunk_id)
        total += sys.getsizeof(chunk)
        total += sys.getsizeof(chunk.token_ids)
        total += sum(sys.getsizeof(token_id) for token_id in chunk.token_ids)
        total += sys.getsizeof(chunk.kv_by_layer)
        total += sum(sys.getsizeof(layer_name) for layer_name in chunk.kv_by_layer)
    return total
