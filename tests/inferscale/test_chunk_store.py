from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest

from inferscale.v1.kv.chunk_store import (
    build_chunk_store,
    close_chunk_store,
    fetch_chunks,
    fetch_device_chunks,
    finalize_chunk_store,
    register_chunks,
    release_chunks,
)
from inferscale.v1.kv.memory_store import GPUMemoryStore
from inferscale.v1.kv.pinned_memory_store import CpuPinnedMemoryStore, _HostUserMemory
from inferscale.v1.kv.packed_memory_store import PackedGPUMemoryStore
from inferscale.v1.types import KVChunk


def _chunk(chunk_id: str, seed: int) -> KVChunk:
    return KVChunk(
        chunk_id=chunk_id,
        token_ids=[seed, seed + 1],
        kv_by_layer={"layer": SimpleNamespace(nbytes=8 * seed)},
        context_ids=(f"turn-{seed}",),
        context_prefix_tokens=seed,
    )


def test_build_chunk_store_selects_backend() -> None:
    store = build_chunk_store("gpu", device="cuda:0", top_k=50, staging_slots=4)
    assert isinstance(store, PackedGPUMemoryStore)
    with pytest.raises(ValueError, match="Unknown chunk store backend"):
        build_chunk_store("pinned", device="cuda:0", top_k=50, staging_slots=4)


def test_finalize_always_builds_lookup_after_payload_packing() -> None:
    store = Mock()
    finalize_chunk_store(store)
    assert [call[0] for call in store.mock_calls] == ["finalize_packed", "finalize_chunk_lookup"]


def test_corpus_lookup_rejects_cpu_target_without_fallback() -> None:
    store = GPUMemoryStore(device="cpu")
    meta = register_chunks(store, {"fact-a": _chunk("fact-a", 1)})
    with pytest.raises(ValueError, match="requires a CUDA device"):
        finalize_chunk_store(store)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        fetch_chunks(store, meta, ["fact-a"])


def test_register_moves_kv_and_returns_metadata_map() -> None:
    store = GPUMemoryStore(device="cuda:0")
    chunks = {"fact-a": _chunk("fact-a", 1), "fact-b": _chunk("fact-b", 2)}

    meta = register_chunks(store, chunks)

    assert set(store.get_all_user_ids()) == {"fact-a", "fact-b"}
    assert meta["fact-a"].token_ids == [1, 2]
    assert meta["fact-a"].kv_by_layer == {}
    assert meta["fact-a"].context_ids == ("turn-1",)
    assert meta["fact-b"].context_prefix_tokens == 2
    assert store.get_user_memory("fact-b").token_ids == [2, 3]
    assert chunks["fact-a"].kv_by_layer == {}
    assert chunks["fact-b"].kv_by_layer == {}


def test_fetch_preserves_order_and_rebuilds_chunks() -> None:
    store = _FakeStagingStore()
    meta = register_chunks(store, {"fact-a": _chunk("fact-a", 1), "fact-b": _chunk("fact-b", 2)})

    fetched = fetch_chunks(store, meta, ["fact-b", "fact-a"])

    assert [chunk.token_ids for chunk in fetched] == [[2, 3], [1, 2]]
    # KV comes from the store, metadata from the map.
    assert fetched[0].kv_by_layer["layer"].nbytes == 16
    assert fetched[0].context_ids == ("turn-2",)
    # Releasing the staged views retains the registered corpus.
    release_chunks(store, ["fact-b", "fact-a"])
    assert set(store.get_all_user_ids()) == {"fact-a", "fact-b"}


def test_fetch_rejects_unknown_ids_and_capacity_overflow() -> None:
    store = _FakeStagingStore()
    meta = register_chunks(store, {"fact-a": _chunk("fact-a", 1)})

    with pytest.raises(RuntimeError, match="no pre-encoded KV chunk"):
        fetch_chunks(store, meta, ["fact-z"])

    store.remove_user_memory("fact-a")
    with pytest.raises(RuntimeError, match="missing from the chunk store"):
        fetch_chunks(store, meta, ["fact-a"])


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
        raise AssertionError("Corpus fetch must use batch GPU lookup.")

    def get_chunk_memories(self, chunk_ids):
        result = []
        for chunk_id in chunk_ids:
            if chunk_id not in self.memories:
                raise ValueError("Retrieved chunk ID has no pre-encoded KV chunk.")
            self.staged.append(chunk_id)
            result.append((chunk_id, self.memories[chunk_id]))
        return result

    def release_staging(self, user_id):
        self.released.append(user_id)

    def get_all_user_ids(self):
        return list(self.memories)

    def remove_user_memory(self, user_id):
        return self.memories.pop(user_id, None) is not None


def test_fetch_release_protocol_on_staging_store() -> None:
    store = _FakeStagingStore()
    meta = register_chunks(
        store, {"fact-a": _chunk("fact-a", 1), "fact-b": _chunk("fact-b", 2), "fact-c": _chunk("fact-c", 3)}
    )

    fetch_chunks(store, meta, ["fact-c", "fact-a"])
    release_chunks(store, ["fact-c", "fact-a"])

    assert store.staged == ["fact-c", "fact-a"]
    assert store.released == ["fact-c", "fact-a"]

    with pytest.raises(RuntimeError, match="staging slots"):
        fetch_chunks(store, meta, ["fact-a", "fact-b", "fact-c", "fact-a"])

    close_chunk_store(store)
    assert store.memories == {}


class _NoTextPayloadLookup(dict):
    def __getitem__(self, key):
        raise AssertionError("Selected corpus payload must be accessed by row.")

    def get(self, key, default=None):
        raise AssertionError("Selected corpus payload must be accessed by row.")


def _lookup_result(rows):
    """Mock only the map API boundary, without simulating CUDA kernels."""
    result = Mock()
    result.detach.return_value.cpu.return_value.tolist.return_value = rows
    return result


def test_gpu_corpus_uses_lookup_rows_and_invalidates_after_mutation(monkeypatch) -> None:
    lookup = Mock(device="cuda:0", nbytes=48)
    lookup.lookup.return_value = _lookup_result([1, 0])
    constructor = Mock(return_value=lookup)
    monkeypatch.setattr("inferscale.v1.kv.memory_store.GPUChunkMap", constructor)
    store = GPUMemoryStore(device="cuda:0")
    meta = register_chunks(store, {"fact-a": _chunk("fact-a", 1), "fact-b": _chunk("fact-b", 2)})
    finalize_chunk_store(store)
    finalize_chunk_store(store)
    constructor.assert_called_once_with(["fact-a", "fact-b"], device="cuda:0")
    store._memories = _NoTextPayloadLookup(store._memories)

    fetched = fetch_chunks(store, meta, ["fact-b", "fact-a"])

    lookup.lookup.assert_called_once_with(["fact-b", "fact-a"])
    assert [chunk.token_ids for chunk in fetched] == [[2, 3], [1, 2]]
    assert store.get_stats()["chunk_map_bytes"] == 48
    store.add_user_memory("fact-c", {}, 0, [])
    lookup.close.assert_called_once()
    assert store._chunk_lookup is None
    assert store._chunk_memories == ()


def test_old_stable_binding_is_rejected_after_corpus_mutation(monkeypatch) -> None:
    lookup = Mock(device="cuda:0", nbytes=48)
    monkeypatch.setattr("inferscale.v1.kv.memory_store.GPUChunkMap", Mock(return_value=lookup))
    store = GPUMemoryStore(device="cuda:0")
    store.add_user_memory("fact-a", {}, 0, [])
    old_map = store.build_device_row_map([(0, "fact-a")])
    store.add_user_memory("fact-b", {}, 0, [])

    with pytest.raises(RuntimeError, match="not bound to the current corpus"):
        store.get_device_chunk_memories(Mock(), old_map)


def test_pinned_corpus_passes_resolved_row_payload_to_staging() -> None:
    # Isolate the GPU map/staging protocol; real CUDA copies are tested below.
    store = CpuPinnedMemoryStore.__new__(CpuPinnedMemoryStore)
    first = _HostUserMemory({}, 1, [11], 4)
    second = _HostUserMemory({}, 1, [22], 4)
    store._host = _NoTextPayloadLookup({"a": first, "b": second})
    store._chunk_host = (("a", first), ("b", second))
    store._chunk_lookup = Mock()
    store._chunk_lookup.lookup.return_value = _lookup_result([1, 0])
    store._lock = threading.Lock()
    store._staged_memory = Mock(side_effect=lambda chunk_id, memory: memory)

    memories = store.get_chunk_memories(["b", "a"])

    assert memories == [("b", second), ("a", first)]
    assert [call.args for call in store._staged_memory.call_args_list] == [("b", second), ("a", first)]


def test_device_fetch_attaches_metadata_without_second_id_lookup() -> None:
    store = Mock(num_staging_slots=3)
    store.get_device_chunk_memories.return_value = [
        ("fact-b", SimpleNamespace(kv_by_layer={"layer": "selected-row"}))
    ]
    stable_ids = Mock()
    stable_ids.numel.return_value = 1
    row_map = object()

    fetched = fetch_device_chunks(store, {"fact-b": _chunk("fact-b", 2)}, stable_ids, row_map)

    store.get_device_chunk_memories.assert_called_once_with(stable_ids, row_map, reverse=True)
    store.get_chunk_memories.assert_not_called()
    store.get_user_memory.assert_not_called()
    assert fetched[0].context_ids == ("turn-2",)
    assert fetched[0].kv_by_layer == {"layer": "selected-row"}


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_cuda_lookup_selects_payloads_without_host_id_lookup(backend) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    store = build_chunk_store(backend, device="cuda:0", top_k=2)
    chunks = {
        chunk_id: KVChunk(chunk_id, [token], {"layer": torch.full((2, 1, 1, 1), token, device="cuda:0")})
        for chunk_id, token in (("a", 11), ("b", 22))
    }
    meta = register_chunks(store, chunks)
    finalize_chunk_store(store)
    row_map = store.build_device_row_map([(0, "b"), (1, "a")])
    payload_attr = "_host" if backend == "cpu" else "_memories"
    original_payloads = getattr(store, payload_attr)
    setattr(store, payload_attr, _NoTextPayloadLookup(original_payloads))
    try:
        fetched = fetch_chunks(store, meta, ["b", "a"])
        assert [chunk.token_ids for chunk in fetched] == [[22], [11]]
        assert [chunk.kv_by_layer["layer"].flatten()[0].item() for chunk in fetched] == [22, 11]
        assert all(chunk.kv_by_layer["layer"].device.type == "cuda" for chunk in fetched)
    finally:
        setattr(store, payload_attr, original_payloads)
        release_chunks(store, ["a", "b"])

    fetched = fetch_device_chunks(
        store, meta, torch.tensor([0, 1], device="cuda:0"), row_map,
    )
    assert [chunk.token_ids for chunk in fetched] == [[11], [22]]
    release_chunks(store, ["a", "b"])
    assert store.get_stats()["chunk_map_device"] == "cuda:0"
    assert store.get_stats()["chunk_map_bytes"] > 0
    close_chunk_store(store)
    assert store.get_stats()["total_gpu_mb"] == 0
