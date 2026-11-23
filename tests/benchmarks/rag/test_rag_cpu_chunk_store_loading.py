from __future__ import annotations

from unittest.mock import Mock

import pytest

from benchmarks.rag import cpu_chunk_store
from benchmarks.rag.cpu_chunk_store import CpuChunkStore
from benchmarks.rag.data_types import RagChunk
from benchmarks.rag.kv_cache import RagChunkCache, RagChunkCacheKey
from inferscale.v1.kv.chunk_map import GPUChunkMap
from inferscale.v1.types import KVChunk


@pytest.fixture
def cache(tmp_path) -> RagChunkCache:
    key = RagChunkCacheKey(
        dataset="multihoprag",
        model="llama",
        dtype="bfloat16",
        chunk_size=1024,
        context_window=5,
        max_position=32768,
        corpus_fingerprint="abc123",
    )
    return RagChunkCache(key=key, directory=tmp_path / "cache")


@pytest.fixture
def events(monkeypatch) -> list[str]:
    happened: list[str] = []

    class Clock:
        def perf_counter(self) -> float:
            happened.append("clock")
            return float(happened.count("clock"))

    def build_map(chunk_ids, *, device):
        happened.append("map")
        return Mock(spec=GPUChunkMap, device=device, nbytes=0)

    def load(path, **_):
        happened.append("load")
        return KVChunk(chunk_id=path.stem, token_ids=[1], kv_by_layer={})

    monkeypatch.setattr(cpu_chunk_store, "time", Clock())
    monkeypatch.setattr(cpu_chunk_store, "GPUChunkMap", build_map)
    monkeypatch.setattr(cpu_chunk_store, "load_chunk", load)
    return happened


def cached_chunks(cache: RagChunkCache, count: int, *, file_bytes: int = 1) -> list[RagChunk]:
    chunks = [RagChunk(f"d1:{index}", "d1", index, [1], "") for index in range(count)]
    for chunk in chunks:
        path = cache.chunk_path(chunk.chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * file_bytes)
    return chunks


def test_load_time_covers_the_chunk_map_and_every_chunk_load(cache, events, rag_log):
    chunks = cached_chunks(cache, 2)

    store = CpuChunkStore(cache, chunks=chunks, device="cuda:0")

    assert events == ["clock", "map", "load", "load", "clock"]
    assert store.load_time_ms == 1000.0
    assert store.stats()["kv_store_load_time_ms"] == 1000.0
    assert rag_log == [
        ("INFO", f"Loading 2 RAG KV chunks (0.0 GiB on disk) into host RAM from {cache.directory}"),
        ("INFO", "Loaded 2/2 KV chunks into host RAM (0.0 GiB)"),
    ]


@pytest.mark.parametrize(
    "chunk_file_bytes,host_memory_bytes,warned",
    [(91, 100, True), (90, 100, False)],
)
def test_chunks_leaving_under_a_tenth_of_host_ram_to_the_engine_and_page_cache_raise_a_warning(
    cache, events, rag_log, monkeypatch, chunk_file_bytes, host_memory_bytes, warned
):
    monkeypatch.setattr(cpu_chunk_store, "total_host_memory_bytes", lambda: host_memory_bytes)
    chunks = cached_chunks(cache, 1, file_bytes=chunk_file_bytes)

    CpuChunkStore(cache, chunks=chunks, device="cuda:0")

    low_memory_warning = (
        "WARNING",
        "RAG KV chunks need about 0.0 GiB of host RAM but this host reports "
        "0.0 GiB total; the cpu store may not fit.",
    )
    if warned:
        assert rag_log[0] == low_memory_warning
    assert [level for level, _ in rag_log].count("WARNING") == int(warned)
