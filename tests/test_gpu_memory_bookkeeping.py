from __future__ import annotations

from types import SimpleNamespace

import pytest

from locomo_jasper_bench.kv.connector_utils import (
    align_to_block_size,
    build_slot_mapping,
    extra_config,
    extract_user_id,
)
from locomo_jasper_bench.kv.gpu_memory_store import GPUMemoryStore


class FakeTensor:
    """Duck-typed stand-in for a torch tensor pinned to one device."""

    def __init__(self, nbytes: int, device: str = "cuda:0") -> None:
        self.nbytes = nbytes
        self.device = device

    def contiguous(self) -> "FakeTensor":
        return self


def test_gpu_memory_store_tracks_tokens_and_bytes_across_users() -> None:
    store = GPUMemoryStore(device="cuda:0")
    store.add_user_memory("alice", {"layer0": FakeTensor(1024), "layer1": FakeTensor(1024)}, 8)
    store.add_user_memory("bob", {"layer0": FakeTensor(2048)}, 4)

    assert sorted(store.get_all_user_ids()) == ["alice", "bob"]
    assert store.get_user_memory("alice").num_tokens == 8
    assert store.get_stats() == {
        "num_users": 2,
        "total_tokens": 12,
        "total_gpu_mb": 4096 / (1024 * 1024),
    }

    assert store.remove_user_memory("alice") is True
    assert store.remove_user_memory("alice") is False
    assert store.get_user_memory("alice") is None
    assert store.get_stats() == {
        "num_users": 1,
        "total_tokens": 4,
        "total_gpu_mb": 2048 / (1024 * 1024),
    }


def test_gpu_memory_store_replaces_existing_user_without_double_counting() -> None:
    store = GPUMemoryStore(device="cuda:0")
    store.add_user_memory("alice", {"layer0": FakeTensor(1024)}, 8, token_ids=[1, 2])
    store.add_user_memory("alice", {"layer0": FakeTensor(512)}, 4, token_ids=[3])

    assert store.get_user_memory("alice").token_ids == [3]
    assert store.get_stats() == {
        "num_users": 1,
        "total_tokens": 4,
        "total_gpu_mb": 512 / (1024 * 1024),
    }


def test_align_to_block_size_floors_to_whole_blocks() -> None:
    assert align_to_block_size(0, 16) == 0
    assert align_to_block_size(15, 16) == 0
    assert align_to_block_size(16, 16) == 16
    assert align_to_block_size(47, 16) == 32


def test_slot_mapping_scatters_tokens_into_paged_blocks() -> None:
    # Blocks 3 and 7 at block_size 4 cover slots 12-15 and 28-31.
    assert build_slot_mapping([3, 7], 4, 6) == [12, 13, 14, 15, 28, 29]
    assert build_slot_mapping([3, 7], 4, 8) == [12, 13, 14, 15, 28, 29, 30, 31]
    assert build_slot_mapping([], 4, 0) == []


def test_slot_mapping_rejects_more_tokens_than_allocated_capacity() -> None:
    with pytest.raises(ValueError, match="exceeds the allocated capacity"):
        build_slot_mapping([3], 4, 5)


def test_extract_user_id_prefers_request_then_sampling_then_metadata() -> None:
    request = SimpleNamespace(user="direct", sampling_params=None, metadata=None)
    assert extract_user_id(request) == "direct"

    request = SimpleNamespace(
        user=None,
        sampling_params=SimpleNamespace(user="sampled"),
        metadata={"user_id": "ignored"},
    )
    assert extract_user_id(request) == "sampled"

    request = SimpleNamespace(user=None, sampling_params=None, metadata={"user_id": "meta"})
    assert extract_user_id(request) == "meta"

    request = SimpleNamespace(user=None, sampling_params=None, metadata=None)
    assert extract_user_id(request, "fallback") == "fallback"
    assert extract_user_id(request) is None


def test_extra_config_uses_getter_then_extra_dict() -> None:
    with_getter = SimpleNamespace(get_from_extra_config=lambda key, default: {"ns": "a"}.get(key, default))
    assert extra_config(with_getter, "ns") == "a"
    assert extra_config(with_getter, "missing", "d") == "d"

    plain = SimpleNamespace(kv_connector_extra_config={"ns": "b"})
    assert extra_config(plain, "ns") == "b"
    assert extra_config(plain, "missing", "d") == "d"
