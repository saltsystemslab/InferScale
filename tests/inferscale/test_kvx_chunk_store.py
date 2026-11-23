from __future__ import annotations

from typing import Any

import pytest

from inferscale.v1.kv import pinned_memory_store
from inferscale.v1.kv.chunk_store import build_chunk_store, register_chunks
from inferscale.v1.types import KVChunk


@pytest.fixture
def pinned_store_requests(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    class RecordingPinnedStore:
        def __init__(self, **kwargs: Any) -> None:
            requests.append(kwargs)

    monkeypatch.setattr(pinned_memory_store, "CpuPinnedMemoryStore", RecordingPinnedStore)
    return requests


@pytest.mark.parametrize(("top_k", "staging_slots", "pool_size"), [(50, 4, 54), (50, 55, 55)])
def test_cpu_store_pool_holds_one_full_fetch_plus_headroom_unless_configured_larger(
    pinned_store_requests: list[dict[str, Any]], top_k: int, staging_slots: int, pool_size: int
) -> None:
    build_chunk_store("cpu", device="cuda:1", top_k=top_k, staging_slots=staging_slots)

    assert pinned_store_requests == [{"device": "cuda:1", "num_staging_slots": pool_size}]


class RegisteredIds:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def add_user_memory(
        self, user_id: str, kv_by_layer: Any, num_tokens: int, token_ids: Any
    ) -> None:
        self.ids.append(user_id)


def test_mismatched_key_registers_none_of_the_chunks() -> None:
    store = RegisteredIds()
    chunks = {
        "fact-a": KVChunk("fact-a", [1], {"layer": object()}),
        "other": KVChunk("fact-b", [2], {"layer": object()}),
    }

    with pytest.raises(ValueError, match="'other' does not match chunk_id 'fact-b'"):
        register_chunks(store, chunks)

    assert store.ids == []
    assert [list(chunk.kv_by_layer) for chunk in chunks.values()] == [["layer"], ["layer"]]
