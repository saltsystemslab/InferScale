"""The corpus KV resident in host RAM, addressed through a CUDA chunk map.

Loaded once from the per-chunk precompute cache into pageable CPU tensors;
composition copies the selected chunks to the GPU per query. There is no
eviction and no answer-time disk I/O, so per-query latencies are uniform from
the first query. Chunk ids resolve to rows on CUDA; only the selected row
numbers return to the host to pick the CPU tensors for composition.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from loguru import logger

from benchmarks.common.system import total_host_memory_bytes
from benchmarks.rag.data_types import RagChunk
from benchmarks.rag.kv_cache import RagChunkCache, load_chunk
from inferscale.v1.kv.chunk_map import GPUChunkMap
from inferscale.v1.kv.chunk_store import chunk_nbytes
from inferscale.v1.types import KVChunk

LOAD_LOG_EVERY = 200
# The chunks must fit in RAM beside the engine and the page cache; above this
# share of the physical total the load is reported as likely to fail.
HOST_MEMORY_WARNING_FRACTION = 0.9
GIB = 1024**3


class CpuChunkStore:
    def __init__(self, cache: RagChunkCache, *, chunks: Sequence[RagChunk], device: str) -> None:
        self._cache = cache
        self._entries: list[KVChunk] = []
        self.resident_bytes = 0
        self.chunk_count = len(chunks)
        cache.require_complete(chunks)
        file_bytes = sum(cache.chunk_path(chunk.chunk_id).stat().st_size for chunk in chunks)
        _warn_if_low_host_memory(file_bytes)
        logger.info(
            "Loading {} RAG KV chunks ({:.1f} GiB on disk) into host RAM from {}",
            len(chunks),
            file_bytes / GIB,
            cache.directory,
        )
        started = time.perf_counter()
        self._chunk_map = GPUChunkMap([chunk.chunk_id for chunk in chunks], device=device)
        try:
            self._load_entries(chunks)
        except Exception:
            self.close()
            raise
        self.load_time_ms = (time.perf_counter() - started) * 1000

    def fetch(self, chunk_ids: Sequence[str]) -> list[KVChunk]:
        try:
            rows = self._chunk_map.lookup(chunk_ids)
        except ValueError as exc:
            raise RuntimeError("Retrieved chunk id is not part of the corpus.") from exc
        return [self._entries[row] for row in rows.detach().cpu().tolist()]

    def stats(self) -> dict[str, Any]:
        return {
            "kv_store_backend": "cpu",
            "kv_store_chunk_count": self.chunk_count,
            "kv_store_resident_bytes": self.resident_bytes,
            "kv_store_load_time_ms": self.load_time_ms,
            "kv_chunk_map_device": str(self._chunk_map.device),
            "kv_chunk_map_bytes": self._chunk_map.nbytes,
        }

    def close(self) -> None:
        self._chunk_map.close()
        self._entries.clear()
        self.resident_bytes = 0

    def _load_entries(self, chunks: Sequence[RagChunk]) -> None:
        for index, chunk in enumerate(chunks, start=1):
            encoded = self._load_entry(chunk)
            self._entries.append(encoded)
            self.resident_bytes += chunk_nbytes(encoded)
            if index % LOAD_LOG_EVERY == 0 or index == len(chunks):
                logger.info(
                    "Loaded {}/{} KV chunks into host RAM ({:.1f} GiB)",
                    index,
                    len(chunks),
                    self.resident_bytes / GIB,
                )

    def _load_entry(self, chunk: RagChunk) -> KVChunk:
        path = self._cache.chunk_path(chunk.chunk_id)
        encoded = load_chunk(
            path,
            expected_meta=self._cache.key.chunk_meta(chunk),
            expected_token_ids=chunk.token_ids,
            device="cpu",
        )
        if encoded is None:
            raise RuntimeError(
                f"RAG KV chunk cache is stale for chunk {chunk.chunk_id} at {path}. "
                "Run the precompute-kv stage with the same model, chunk size, and "
                "context window first."
            )
        return encoded


def _warn_if_low_host_memory(expected_bytes: int) -> None:
    total = total_host_memory_bytes()
    if total is not None and expected_bytes > HOST_MEMORY_WARNING_FRACTION * total:
        logger.warning(
            "RAG KV chunks need about {:.1f} GiB of host RAM but this host reports "
            "{:.1f} GiB total; the cpu store may not fit.",
            expected_bytes / GIB,
            total / GIB,
        )
