from __future__ import annotations

import gc
import logging
import weakref
from types import SimpleNamespace
from typing import Any

import pytest

from inferscale.v1.kv import chunk_lookup
from inferscale.v1.kv.chunk_store import register_chunks
from inferscale.v1.kv.packed_memory_store import PackedGPUMemoryStore
from inferscale.v1.types import KVChunk

torch = pytest.importorskip("torch")

LAYERS = ("layer-a", "layer-b")
MB = 1024 * 1024


class RowMap:
    pass


class ListBackedChunkMap:
    device = "cpu"
    nbytes = 48

    def __init__(self, chunk_ids: Any, *, device: str) -> None:
        self.chunk_ids = list(chunk_ids)

    def lookup(self, chunk_ids: Any) -> Any:
        rows = [self.chunk_ids.index(chunk_id) for chunk_id in chunk_ids]
        return SimpleNamespace(
            detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: rows))
        )

    def stable_id_row_map(self, bindings: Any) -> RowMap:
        return RowMap()

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def list_backed_chunk_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunk_lookup, "GPUChunkMap", ListBackedChunkMap)


def kv(seed: int, tokens: int, layers: tuple[str, ...] = LAYERS) -> dict[str, Any]:
    first = torch.arange(seed, seed + 2 * tokens, dtype=torch.float32).reshape(2, tokens, 1, 1)
    return {layer: first + 100 * index for index, layer in enumerate(layers)}


def store_with_two_chunks() -> tuple[PackedGPUMemoryStore, dict[str, dict[str, Any]]]:
    store = PackedGPUMemoryStore(device="cpu")
    sources = {"fact-a": kv(10, 2), "fact-b": kv(30, 3)}
    store.add_user_memory("fact-a", dict(sources["fact-a"]), 2, token_ids=[1, 2])
    store.add_user_memory("fact-b", dict(sources["fact-b"]), 3, token_ids=[3, 4, 5])
    return store, sources


def payloads(store: PackedGPUMemoryStore) -> dict[str, dict[str, Any]]:
    return {
        chunk_id: memory.kv_by_layer
        for chunk_id, memory in store.get_chunk_memories(store.get_all_user_ids())
    }


def share_one_slab(first: Any, second: Any) -> bool:
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def test_packing_turns_every_payload_into_a_view_of_one_slab_per_layer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, sources = store_with_two_chunks()
    assert store.get_stats() == {
        "num_users": 2,
        "total_tokens": 5,
        "total_gpu_mb": 80 / MB,
        "gpu_chunk_map_mb": 0.0,
        "chunk_map_bytes": 0,
        "chunk_map_device": None,
    }

    with caplog.at_level(logging.INFO, logger="inferscale.v1.kv.packed_memory_store"):
        caplog.clear()
        store.finalize_packed()
        store.finalize_packed()

    assert [record.getMessage() for record in caplog.records] == [
        "Packed 2 GPU memory chunks across 2 layers."
    ]
    packed = payloads(store)
    for layer in LAYERS:
        assert torch.equal(packed["fact-a"][layer], sources["fact-a"][layer])
        assert torch.equal(packed["fact-b"][layer], sources["fact-b"][layer])
        assert share_one_slab(packed["fact-a"][layer], packed["fact-b"][layer])
        assert not share_one_slab(packed["fact-a"][layer], sources["fact-a"][layer])
    assert list(packed["fact-a"]) == list(LAYERS)
    slab_bytes, index_bytes = 2 * 40, 5 * 8 + 2 * 8 + 2 * 8
    assert store.get_stats() == {
        "num_users": 2,
        "total_tokens": 5,
        "total_gpu_mb": (slab_bytes + index_bytes + 48) / MB,
        "gpu_chunk_map_mb": 48 / MB,
        "chunk_map_bytes": 48,
        "chunk_map_device": "cpu",
    }


def test_payloads_point_at_the_slabs_even_when_the_index_tensors_cannot_be_allocated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, sources = store_with_two_chunks()

    def out_of_memory(*args: Any, **kwargs: Any) -> Any:
        raise MemoryError("no room for the index tensors")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", out_of_memory)
        with pytest.raises(MemoryError, match="no room for the index tensors"):
            store.finalize_packed()

    half_packed = payloads(store)
    for layer in LAYERS:
        assert torch.equal(half_packed["fact-a"][layer], sources["fact-a"][layer])
        assert torch.equal(half_packed["fact-b"][layer], sources["fact-b"][layer])
        assert share_one_slab(half_packed["fact-a"][layer], half_packed["fact-b"][layer])
    assert store.get_stats()["total_gpu_mb"] == (80 + 48) / MB

    store.finalize_packed()
    assert store.get_stats()["total_gpu_mb"] == (80 + 72 + 48) / MB
    assert torch.equal(payloads(store)["fact-b"]["layer-b"], sources["fact-b"]["layer-b"])


def test_packing_frees_the_tensors_that_registered_chunks_handed_over() -> None:
    store = PackedGPUMemoryStore(device="cpu")
    chunks = {
        chunk_id: KVChunk(chunk_id, [token], kv(seed, 1))
        for chunk_id, token, seed in (("fact-a", 1, 10), ("fact-b", 2, 30))
    }
    pre_pack_tensors = [
        weakref.ref(tensor) for chunk in chunks.values() for tensor in chunk.kv_by_layer.values()
    ]

    register_chunks(store, chunks)
    assert all(tensor() is not None for tensor in pre_pack_tensors)

    store.finalize_packed()
    gc.collect()

    assert all(tensor() is None for tensor in pre_pack_tensors)
    assert payloads(store)["fact-b"]["layer-a"].tolist() == [[[[30.0]]], [[[31.0]]]]


def test_selected_layers_are_gathered_on_access_not_copied_up_front(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, sources = store_with_two_chunks()
    store.finalize_packed()
    row_map = store.build_device_row_map([(0, "fact-b"), (1, "fact-a")])
    monkeypatch.setattr(
        chunk_lookup, "device_chunk_rows", lambda stable_ids, id_to_row, **options: stable_ids
    )

    selected = store.select_device_ids(torch.tensor([1, 0]), row_map, reverse=True)

    assert (selected.token_ids, selected.num_tokens) == ([3, 4, 5, 1, 2], 5)
    layers = selected.kv_by_layer
    assert (list(layers), len(layers), layers.keys()) == (list(LAYERS), 2, LAYERS)
    for layer in LAYERS:
        expected = torch.cat([sources["fact-b"][layer], sources["fact-a"][layer]], dim=1)
        assert torch.equal(layers[layer], expected)
    assert layers["layer-a"] is not layers["layer-a"]
    with pytest.raises(KeyError, match="layer-z"):
        layers["layer-z"]

    store.remove_user_memory("fact-a")
    assert torch.equal(layers["layer-b"][:, :3], sources["fact-b"]["layer-b"])
    store.remove_user_memory("fact-b")
    with pytest.raises(RuntimeError, match="'layer-a' is unavailable after store close"):
        layers["layer-a"]
