from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from inferscale.v1.kv.packed_memory_store import (  # noqa: E402
    DeviceChunkSelectionError,
    PackedGPUMemoryStore,
)


def _kv(seed: int, length: int) -> dict[str, object]:
    first = torch.arange(
        seed,
        seed + 2 * length,
        dtype=torch.float32,
        device="cuda:0",
    ).reshape(2, length, 1, 1)
    second = (first + 100).clone()
    return {"layer-a": first, "layer-b": second}


def test_packed_selection_matches_string_lookup_with_variable_lengths() -> None:
    store = PackedGPUMemoryStore(device="cuda:0")
    fact_a = _kv(10, 2)
    fact_b = _kv(30, 3)
    store.add_user_memory("fact-a", fact_a, 2, token_ids=[1, 2])
    store.add_user_memory("fact-b", fact_b, 3, token_ids=[3, 4, 5])
    store.finalize_packed()

    row_map = store.build_device_row_map(((0, "fact-a"), (1, "fact-b")))
    selected = store.select_device_ids(
        torch.tensor([0, 1], dtype=torch.int32, device="cuda:0"),
        row_map,
        reverse=True,
    )

    assert selected.token_ids == [3, 4, 5, 1, 2]
    for layer_name in ("layer-a", "layer-b"):
        expected = torch.cat([fact_b[layer_name], fact_a[layer_name]], dim=1)
        assert torch.equal(selected.kv_by_layer[layer_name], expected)

    [(chunk_id, fetched_a)] = store.get_chunk_memories(["fact-a"])
    assert chunk_id == "fact-a"
    assert fetched_a.token_ids == [1, 2]
    assert torch.equal(fetched_a.kv_by_layer["layer-a"], fact_a["layer-a"])
    assert row_map.device.type == "cuda"
    assert store.get_stats()["chunk_map_device"] == "cuda:0"


def test_packed_selection_uses_explicit_stable_id_to_row_binding() -> None:
    store = PackedGPUMemoryStore(device="cuda:0")
    fact_a = _kv(10, 1)
    fact_b = _kv(20, 1)
    store.add_user_memory("fact-a", fact_a, 1, token_ids=[1])
    store.add_user_memory("fact-b", fact_b, 1, token_ids=[2])
    store.finalize_packed()

    row_map = store.build_device_row_map(((0, "fact-b"), (1, "fact-a")))
    selected = store.select_device_ids(
        torch.tensor([0, 1], dtype=torch.int32, device="cuda:0"),
        row_map,
        reverse=True,
    )

    assert selected.token_ids == [1, 2]
    expected = torch.cat([fact_a["layer-a"], fact_b["layer-a"]], dim=1)
    assert torch.equal(selected.kv_by_layer["layer-a"], expected)


@pytest.mark.parametrize(
    "stable_ids",
    [
        [-1],
        [0, -1],
        [0, 0],
        [7],
        [],
    ],
)
def test_packed_selection_rejects_padded_duplicate_invalid_and_empty_ids(
    stable_ids: list[int],
) -> None:
    store = PackedGPUMemoryStore(device="cuda:0")
    store.add_user_memory("fact-a", _kv(10, 1), 1, token_ids=[1])
    store.finalize_packed()
    row_map = store.build_device_row_map(((0, "fact-a"),))

    with pytest.raises(DeviceChunkSelectionError):
        store.select_device_ids(
            torch.tensor(stable_ids, dtype=torch.int32, device="cuda:0"),
            row_map,
            reverse=True,
        )


def test_packed_store_freezes_registration_and_validates_bindings() -> None:
    store = PackedGPUMemoryStore(device="cuda:0")
    store.add_user_memory("fact-a", _kv(10, 1), 1, token_ids=[1])
    store.finalize_packed()

    with pytest.raises(RuntimeError, match="after.*finalized"):
        store.add_user_memory("fact-b", _kv(20, 1), 1, token_ids=[2])
    with pytest.raises(ValueError, match="no pre-encoded KV chunk"):
        store.build_device_row_map(((0, "fact-missing"),))
    with pytest.raises(ValueError, match="contiguous"):
        store.build_device_row_map(((1, "fact-a"),))


def test_selected_layer_view_does_not_retain_slabs_after_store_close() -> None:
    store = PackedGPUMemoryStore(device="cuda:0")
    store.add_user_memory("fact-a", _kv(10, 1), 1, token_ids=[1])
    store.finalize_packed()
    row_map = store.build_device_row_map(((0, "fact-a"),))
    selected = store.select_device_ids(
        torch.tensor([0], dtype=torch.int32, device="cuda:0"),
        row_map,
        reverse=True,
    )

    assert store.remove_user_memory("fact-a") is True
    assert store.get_stats()["total_gpu_mb"] == 0
    with pytest.raises(RuntimeError, match="unavailable after store close"):
        selected.kv_by_layer["layer-a"]


def test_packed_selection_rejects_cpu_ids_and_stale_bindings() -> None:
    store = PackedGPUMemoryStore(device="cuda:0")
    store.add_user_memory("fact-a", _kv(10, 1), 1, token_ids=[1])
    store.add_user_memory("fact-b", _kv(20, 1), 1, token_ids=[2])
    store.finalize_packed()
    row_map = store.build_device_row_map(((0, "fact-a"), (1, "fact-b")))

    with pytest.raises(DeviceChunkSelectionError, match="CUDA device"):
        store.select_device_ids(torch.tensor([0]), row_map, reverse=True)
    with pytest.raises(DeviceChunkSelectionError, match="not bound"):
        store.select_device_ids(torch.tensor([0], device="cuda:0"), row_map.cpu(), reverse=True)

    store.remove_user_memory("fact-a")
    with pytest.raises(RuntimeError, match="invalid after chunk removal"):
        store.select_device_ids(torch.tensor([1], device="cuda:0"), row_map, reverse=True)
