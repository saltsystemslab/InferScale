from __future__ import annotations

from types import SimpleNamespace

import pytest

from locomo_jasper_bench.kv.chunk_store import (
    build_chunk_store,
    close_chunk_store,
    fetch_fact_chunks,
    finalize_chunk_store,
    register_fact_chunks,
    release_fact_chunks,
)
from locomo_jasper_bench.kv.gpu_memory_store import GPUMemoryStore
from locomo_jasper_bench.kv.packed_gpu_memory_store import PackedGPUMemoryStore
from locomo_jasper_bench.kv.types import EncodedChunk


def _chunk(seed: int) -> EncodedChunk:
    return EncodedChunk(
        token_ids=[seed, seed + 1],
        kv_by_layer={"layer": SimpleNamespace(nbytes=8 * seed)},
        context_turn_ids=(f"turn-{seed}",),
        context_prefix_tokens=seed,
    )


def test_build_chunk_store_selects_backend() -> None:
    store = build_chunk_store("gpu", device="cuda:0", top_k=50, staging_slots=4)
    assert isinstance(store, GPUMemoryStore)
    packed = build_chunk_store(
        "gpu",
        device="cuda:0",
        top_k=50,
        staging_slots=4,
        device_selection=True,
    )
    assert isinstance(packed, PackedGPUMemoryStore)
    with pytest.raises(ValueError, match="Unknown chunk store backend"):
        build_chunk_store("pinned", device="cuda:0", top_k=50, staging_slots=4)


def test_finalize_is_a_noop_for_the_ordinary_gpu_store() -> None:
    store = GPUMemoryStore(device="cuda:0")

    finalize_chunk_store(store)

    assert store.get_stats()["num_users"] == 0


def test_register_moves_kv_and_returns_metadata_map() -> None:
    store = GPUMemoryStore(device="cuda:0")
    chunks = {"fact-a": _chunk(1), "fact-b": _chunk(2)}

    meta = register_fact_chunks(store, chunks)

    assert set(store.get_all_user_ids()) == {"fact-a", "fact-b"}
    assert meta["fact-a"].token_ids == [1, 2]
    assert meta["fact-a"].kv_by_layer == {}
    assert meta["fact-a"].context_turn_ids == ("turn-1",)
    assert meta["fact-b"].context_prefix_tokens == 2
    assert store.get_user_memory("fact-b").token_ids == [2, 3]
    assert chunks["fact-a"].kv_by_layer == {}
    assert chunks["fact-b"].kv_by_layer == {}


def test_fetch_preserves_order_and_rebuilds_chunks() -> None:
    store = GPUMemoryStore(device="cuda:0")
    meta = register_fact_chunks(store, {"fact-a": _chunk(1), "fact-b": _chunk(2)})

    fetched = fetch_fact_chunks(store, meta, ["fact-b", "fact-a"])

    assert [chunk.token_ids for chunk in fetched] == [[2, 3], [1, 2]]
    # KV comes from the store, metadata from the map.
    assert fetched[0].kv_by_layer["layer"].nbytes == 16
    assert fetched[0].context_turn_ids == ("turn-2",)
    # Releasing on the gpu store is a no-op; the corpus stays registered.
    release_fact_chunks(store, ["fact-b", "fact-a"])
    assert set(store.get_all_user_ids()) == {"fact-a", "fact-b"}


def test_fetch_rejects_unknown_ids_and_capacity_overflow() -> None:
    store = GPUMemoryStore(device="cuda:0")
    meta = register_fact_chunks(store, {"fact-a": _chunk(1)})

    with pytest.raises(RuntimeError, match="no pre-encoded KV chunk"):
        fetch_fact_chunks(store, meta, ["fact-z"])

    store.remove_user_memory("fact-a")
    with pytest.raises(RuntimeError, match="missing from the chunk store"):
        fetch_fact_chunks(store, meta, ["fact-a"])


class _FakeStagingStore:
    """CpuPinnedMemoryStore stand-in recording the stage/release protocol."""

    num_staging_slots = 3

    def __init__(self) -> None:
        self.memories: dict[str, SimpleNamespace] = {}
        self.staged: list[str] = []
        self.released: list[str] = []

    def add_user_memory(self, user_id, kv_by_layer, num_tokens, token_ids=None):
        self.memories[user_id] = SimpleNamespace(
            kv_by_layer=dict(kv_by_layer), num_tokens=num_tokens, token_ids=token_ids
        )

    def get_user_memory(self, user_id):
        memory = self.memories.get(user_id)
        if memory is not None:
            self.staged.append(user_id)
        return memory

    def release_staging(self, user_id):
        self.released.append(user_id)

    def get_all_user_ids(self):
        return list(self.memories)

    def remove_user_memory(self, user_id):
        return self.memories.pop(user_id, None) is not None


def test_fetch_release_protocol_on_staging_store() -> None:
    store = _FakeStagingStore()
    meta = register_fact_chunks(
        store, {"fact-a": _chunk(1), "fact-b": _chunk(2), "fact-c": _chunk(3)}
    )

    fetch_fact_chunks(store, meta, ["fact-c", "fact-a"])
    release_fact_chunks(store, ["fact-c", "fact-a"])

    assert store.staged == ["fact-c", "fact-a"]
    assert store.released == ["fact-c", "fact-a"]

    with pytest.raises(RuntimeError, match="staging slots"):
        fetch_fact_chunks(store, meta, ["fact-a", "fact-b", "fact-c", "fact-a"])

    close_chunk_store(store)
    assert store.memories == {}
