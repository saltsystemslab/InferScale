from __future__ import annotations

from types import SimpleNamespace

import pytest

vllm = pytest.importorskip("vllm")

from locomo_jasper_bench.kv import gpu_connector
from locomo_jasper_bench.kv.connector_metadata import MemoryConnectorMetadata
from locomo_jasper_bench.kv.gpu_connector import (
    MemoryKVConnector,
    reset_load_stats,
    snapshot_load_stats,
)

torch = pytest.importorskip("torch")


class _FakeStore:
    def __init__(self, memories: dict[str, object]) -> None:
        self._memories = memories

    def get_user_memory(self, user_id: str):
        return self._memories.get(user_id)


def _connector(memories: dict[str, object], loads: list) -> MemoryKVConnector:
    connector = MemoryKVConnector.__new__(MemoryKVConnector)
    connector._memory_store = _FakeStore(memories)
    metadata = MemoryConnectorMetadata()
    for load in loads:
        metadata.loads.append(load)
    if hasattr(connector, "bind_connector_metadata"):
        connector.bind_connector_metadata(metadata)
    else:
        connector._connector_metadata = metadata
    return connector


def _load(user_id: str = "user-0", num_tokens: int = 16, skip_tokens: int = 0):
    return SimpleNamespace(
        user_id=user_id,
        num_tokens=num_tokens,
        skip_tokens=skip_tokens,
        request_id=f"req-{user_id}",
        slot_mapping=torch.arange(num_tokens),
    )


def _forward_context(attn_metadata, layers=None):
    return SimpleNamespace(
        attn_metadata=attn_metadata,
        no_compile_layers=layers or {},
    )


def test_missing_attention_metadata_raises() -> None:
    connector = _connector({}, [_load()])

    with pytest.raises(RuntimeError, match="no attention metadata"):
        connector.start_load_kv(_forward_context(attn_metadata=None))


def test_missing_user_memory_raises() -> None:
    connector = _connector({}, [_load("user-missing")])

    with pytest.raises(RuntimeError, match="user-missing was not found"):
        connector.start_load_kv(_forward_context(attn_metadata=object()))


def test_memory_without_layer_tensors_raises() -> None:
    memory = SimpleNamespace(kv_by_layer={})
    connector = _connector({"user-0": memory}, [_load("user-0")])

    with pytest.raises(RuntimeError, match="has no layer tensors"):
        connector.start_load_kv(_forward_context(attn_metadata=object()))


def test_missing_layer_raises() -> None:
    src_kv = torch.zeros((2, 16, 1, 4))
    memory = SimpleNamespace(kv_by_layer={"layer.0": src_kv})
    layers = {"layer.1": SimpleNamespace(kv_cache=torch.zeros((1, 2, 16, 1, 4)))}
    connector = _connector({"user-0": memory}, [_load("user-0")])

    with pytest.raises(RuntimeError, match="layer.1 not found"):
        connector.start_load_kv(_forward_context(attn_metadata=object(), layers=layers))


def test_no_injectable_layers_raises() -> None:
    src_kv = torch.zeros((2, 16, 1, 4))
    memory = SimpleNamespace(kv_by_layer={"layer.0": src_kv})
    layers = {"layer.0": SimpleNamespace(kv_cache=None)}
    connector = _connector({"user-0": memory}, [_load("user-0")])

    with pytest.raises(RuntimeError, match="No KV cache layers were injected"):
        connector.start_load_kv(_forward_context(attn_metadata=object(), layers=layers))


def test_successful_load_increments_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    src_kv = torch.zeros((2, 16, 1, 4))
    memory = SimpleNamespace(kv_by_layer={"layer.0": src_kv})
    layers = {"layer.0": SimpleNamespace(kv_cache=torch.zeros((1, 2, 16, 1, 4)))}
    connector = _connector({"user-0": memory}, [_load("user-0", num_tokens=16)])
    monkeypatch.setattr(
        MemoryKVConnector, "_inject_kv_into_layer", lambda self, **kwargs: None
    )

    reset_load_stats()
    connector.start_load_kv(_forward_context(attn_metadata=object(), layers=layers))
    stats = snapshot_load_stats()

    assert stats["requests_loaded"] == 1
    assert stats["tokens_loaded"] == 16
    assert stats["requests_covered"] == 1


def test_reset_load_stats_clears_counters() -> None:
    gpu_connector._record_load(8, "req-a")
    gpu_connector._record_native_covered("req-b")
    reset_load_stats()

    assert snapshot_load_stats() == {
        "requests_loaded": 0,
        "tokens_loaded": 0,
        "requests_covered": 0,
    }


def test_requests_covered_unions_loaded_and_native_request_ids() -> None:
    reset_load_stats()
    gpu_connector._record_load(8, "req-a")
    gpu_connector._record_load(8, "req-a")  # preemption re-load: same request
    gpu_connector._record_native_covered("req-b")

    stats = snapshot_load_stats()
    assert stats["requests_loaded"] == 2
    assert stats["requests_covered"] == 2


class _FakeStagingStore(_FakeStore):
    """Fake store exposing the pinned-host staging protocol."""

    def __init__(self, memories: dict[str, object], num_staging_slots: int = 4) -> None:
        super().__init__(memories)
        self.num_staging_slots = num_staging_slots
        self.calls: list[tuple[str, str]] = []

    def prefetch_user_to_gpu(self, user_id: str) -> bool:
        self.calls.append(("prefetch", user_id))
        return user_id in self._memories

    def release_staging(self, user_id: str) -> None:
        self.calls.append(("release", user_id))


def test_staging_store_gets_prefetch_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    src_kv = torch.zeros((2, 16, 1, 4))
    memory = SimpleNamespace(kv_by_layer={"layer.0": src_kv})
    layers = {"layer.0": SimpleNamespace(kv_cache=torch.zeros((1, 2, 16, 1, 4)))}
    connector = _connector({"user-0": memory}, [_load("user-0")])
    store = _FakeStagingStore({"user-0": memory})
    connector._memory_store = store
    monkeypatch.setattr(
        MemoryKVConnector, "_inject_kv_into_layer", lambda self, **kwargs: None
    )

    reset_load_stats()
    connector.start_load_kv(_forward_context(attn_metadata=object(), layers=layers))

    # Release fires once inside the loop and once in the idempotent
    # finally sweep that covers the error path.
    assert store.calls == [
        ("prefetch", "user-0"),
        ("release", "user-0"),
        ("release", "user-0"),
    ]


def test_empty_loads_touch_nothing() -> None:
    connector = _connector({}, [])
    store = _FakeStagingStore({})
    connector._memory_store = store

    # Decode-only steps must be zero-cost: no store calls, and no
    # attn-metadata requirement either.
    connector.start_load_kv(_forward_context(attn_metadata=None))

    assert store.calls == []


def test_prefetch_window_slides_with_staging_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    src_kv = torch.zeros((2, 16, 1, 4))
    layers = {"layer.0": SimpleNamespace(kv_cache=torch.zeros((1, 2, 16, 1, 4)))}
    users = [f"user-{i}" for i in range(5)]
    memories = {user: SimpleNamespace(kv_by_layer={"layer.0": src_kv}) for user in users}
    connector = _connector(memories, [_load(user) for user in users])
    store = _FakeStagingStore(memories, num_staging_slots=2)
    connector._memory_store = store
    monkeypatch.setattr(
        MemoryKVConnector, "_inject_kv_into_layer", lambda self, **kwargs: None
    )

    reset_load_stats()
    connector.start_load_kv(_forward_context(attn_metadata=object(), layers=layers))

    prefetches = [user for kind, user in store.calls if kind == "prefetch"]
    assert prefetches == users
    # Never more than num_staging_slots staged at once: prefetch of user k+2
    # must come after the release of user k.
    for index, user in enumerate(users[2:], start=2):
        release_pos = store.calls.index(("release", users[index - 2]))
        prefetch_pos = store.calls.index(("prefetch", user))
        assert release_pos < prefetch_pos


def test_staging_release_runs_even_when_load_fails() -> None:
    connector = _connector({}, [_load("user-missing")])
    store = _FakeStagingStore({})
    connector._memory_store = store

    with pytest.raises(RuntimeError, match="user-missing was not found"):
        connector.start_load_kv(_forward_context(attn_metadata=object()))

    assert ("release", "user-missing") in store.calls
