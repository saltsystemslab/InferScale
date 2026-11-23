from __future__ import annotations

import contextlib
import logging
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from inferscale.v1.kv.pinned_memory_store import CpuPinnedMemoryStore

MB = 1024 * 1024
BYTES_PER_VALUE = 4


class FakeTensor:
    def __init__(self, journal: list[str], shape: tuple[int, ...], dtype: str, home: str) -> None:
        self.journal = journal
        self.shape = shape
        self.dtype = dtype
        self.home = home
        self.copied_from: FakeTensor | None = None
        self.nbytes = BYTES_PER_VALUE
        for size in shape:
            self.nbytes *= size

    def detach(self) -> FakeTensor:
        return self

    def copy_(self, source: FakeTensor, non_blocking: bool) -> None:
        self.copied_from = source
        blocking = "non-blocking" if non_blocking else "blocking"
        self.journal.append(f"{blocking} copy {source.home} -> {self.home}")


class FakeCudaEvent:
    def __init__(self, cuda: FakeCuda, enable_timing: bool = False) -> None:
        self.cuda = cuda
        self.name = "timing event" if enable_timing else "layer event"
        self.recorded_at_ms = 0.0

    def record(self, stream: str) -> None:
        self.cuda.clock_ms += 1.0
        self.recorded_at_ms = self.cuda.clock_ms
        self.cuda.journal.append(f"record {self.name} on {stream}")

    def synchronize(self) -> None:
        self.cuda.journal.append(f"synchronize {self.name}")

    def elapsed_time(self, end: FakeCudaEvent) -> float:
        return end.recorded_at_ms - self.recorded_at_ms


class FakeCuda:
    def __init__(self, journal: list[str]) -> None:
        self.journal = journal
        self.available = True
        self.clock_ms = 0.0

    def is_available(self) -> bool:
        return self.available

    @contextlib.contextmanager
    def device(self, device: str) -> Iterator[None]:
        yield

    def Stream(self, device: str) -> str:  # noqa: N802
        return "copy stream"

    @contextlib.contextmanager
    def stream(self, stream: str) -> Iterator[None]:
        self.journal.append(f"enter {stream}")
        yield
        self.journal.append(f"leave {stream}")

    def Event(self, enable_timing: bool = False) -> FakeCudaEvent:  # noqa: N802
        return FakeCudaEvent(self, enable_timing)


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    journal: list[str] = []
    torch = types.ModuleType("torch")
    torch.journal = journal  # type: ignore[attr-defined]
    torch.cuda = FakeCuda(journal)  # type: ignore[attr-defined]
    torch.device = lambda name: name  # type: ignore[attr-defined]

    def empty(
        shape: tuple[int, ...], dtype: str, pin_memory: bool = False, device: Any = None
    ) -> FakeTensor:
        home = "pinned host" if pin_memory else str(device)
        journal.append(f"allocate {home}")
        return FakeTensor(journal, shape, dtype, home)

    torch.empty = empty  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def gpu_kv(journal: list[str], tokens: int, layers: int = 2) -> dict[str, FakeTensor]:
    return {
        f"layer.{index}": FakeTensor(journal, (2, tokens, 1, 8), "float16", "encoder gpu")
        for index in range(layers)
    }


def store_with(
    fake_torch: Any, users: dict[str, int], num_staging_slots: int = 2
) -> CpuPinnedMemoryStore:
    store = CpuPinnedMemoryStore(device="cuda:0", num_staging_slots=num_staging_slots)
    for user_id, tokens in users.items():
        store.add_user_memory(
            user_id, gpu_kv(fake_torch.journal, tokens), tokens, token_ids=list(range(tokens))
        )
    fake_torch.journal.clear()
    return store


def payload_bytes(tokens: int, layers: int = 2) -> int:
    return layers * 2 * tokens * 8 * BYTES_PER_VALUE


def test_registration_copies_every_layer_to_pinned_host_memory_and_waits_for_it(
    fake_torch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    store = CpuPinnedMemoryStore(num_staging_slots=2)
    source = gpu_kv(fake_torch.journal, tokens=3)

    with caplog.at_level(logging.DEBUG, logger="inferscale.v1.kv.pinned_memory_store"):
        caplog.clear()
        store.add_user_memory("u0", source, num_tokens=3, token_ids=(1, 2, 3))

    assert fake_torch.journal == [
        "allocate pinned host",
        "blocking copy encoder gpu -> pinned host",
        "allocate pinned host",
        "blocking copy encoder gpu -> pinned host",
    ]
    host = store._host["u0"]
    assert list(host.kv_by_layer_host) == ["layer.0", "layer.1"]
    assert host.kv_by_layer_host["layer.1"].copied_from is source["layer.1"]
    assert host.kv_by_layer_host["layer.1"].shape == (2, 3, 1, 8)
    assert host.kv_by_layer_host["layer.1"].dtype == "float16"
    assert (host.num_tokens, host.token_ids, host.nbytes) == (3, [1, 2, 3], payload_bytes(3))
    megabytes = payload_bytes(3) / MB
    assert [record.getMessage() for record in caplog.records] == [
        f"Stored pinned-host memory for user u0: 3 tokens, 2 layers, {megabytes:.1f} MB"
    ]


def test_prefetch_copies_every_layer_on_the_copy_stream_between_the_timing_events(
    fake_torch: Any,
) -> None:
    store = store_with(fake_torch, {"u0": 3})

    assert store.prefetch_user_to_gpu("missing") is False
    assert store.prefetch_user_to_gpu("u0") is True

    one_layer = [
        "allocate cuda:0",
        "non-blocking copy pinned host -> cuda:0",
        "record layer event on copy stream",
    ]
    assert fake_torch.journal == [
        "enter copy stream",
        "record timing event on copy stream",
        *one_layer,
        *one_layer,
        "record timing event on copy stream",
        "leave copy stream",
    ]

    fake_torch.journal.clear()
    assert store.prefetch_user_to_gpu("u0") is True
    assert fake_torch.journal == []


def test_release_waits_for_the_copies_and_records_one_transfer(fake_torch: Any) -> None:
    store = store_with(fake_torch, {"u0": 3})
    store.prefetch_user_to_gpu("u0")
    fake_torch.journal.clear()

    store.release_staging("u0")
    store.release_staging("u0")
    store.release_staging("never-staged")

    assert fake_torch.journal == ["synchronize timing event"]
    assert store._slots == {}
    assert store.transfer_totals() == {
        "total_transfers": 1.0,
        "total_bytes_transferred": float(payload_bytes(3)),
        "total_h2d_latency_ms": 3.0,
        "total_staging_stall_ms": 0.0,
    }


def test_reregistering_a_staged_memory_records_its_transfer_with_the_previous_sizes(
    fake_torch: Any,
) -> None:
    store = store_with(fake_torch, {"u0": 3})
    store.prefetch_user_to_gpu("u0")

    store.add_user_memory("u0", gpu_kv(fake_torch.journal, tokens=50), 50, token_ids=[7] * 50)

    assert store._slots == {}
    assert store.get_bench_summary()["total_bytes_transferred"] == payload_bytes(3)
    assert store._host["u0"].nbytes == payload_bytes(50)


def test_removing_a_staged_memory_records_its_transfer_with_its_real_sizes(
    fake_torch: Any,
) -> None:
    store = store_with(fake_torch, {"u0": 3, "u1": 5})
    store.prefetch_user_to_gpu("u1")

    assert store.remove_user_memory("u1") is True
    assert store.remove_user_memory("u1") is False
    assert store.remove_user_memory("u0") is True

    assert store.get_all_user_ids() == []
    assert store.get_bench_summary()["total_transfers"] == 1
    assert store.get_bench_summary()["total_bytes_transferred"] == payload_bytes(5)


def test_full_pool_evicts_its_oldest_memory_and_warns_that_it_is_undersized(
    fake_torch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    store = store_with(fake_torch, {"u0": 3, "u1": 3, "u2": 3}, num_staging_slots=2)

    with caplog.at_level(logging.WARNING, logger="inferscale.v1.kv.pinned_memory_store"):
        caplog.clear()
        store.prefetch_user_to_gpu("u0")
        store.prefetch_user_to_gpu("u1")
        store.get_user_memory("u0")
        assert caplog.records == []
        store.prefetch_user_to_gpu("u2")

    assert list(store._slots) == ["u1", "u2"]
    assert [record.getMessage() for record in caplog.records] == [
        "Evicting staging slot for user u0; consider raising num_staging_slots"
    ]
    assert store.get_bench_summary()["total_transfers"] == 1


def test_stats_count_the_staging_pool_as_gpu_memory_and_the_payload_as_host_memory(
    fake_torch: Any,
) -> None:
    store = store_with(fake_torch, {"u0": 3, "u1": 5})
    host_mb = (payload_bytes(3) + payload_bytes(5)) / MB
    assert store.get_stats() == {
        "num_users": 2,
        "total_tokens": 8,
        "total_gpu_mb": 0.0,
        "gpu_staging_mb": 0.0,
        "gpu_chunk_map_mb": 0.0,
        "chunk_map_bytes": 0,
        "chunk_map_device": None,
        "total_host_mb": host_mb,
    }

    store.prefetch_user_to_gpu("u1")

    staged_mb = payload_bytes(5) / MB
    stats = store.get_stats()
    assert (stats["total_gpu_mb"], stats["gpu_staging_mb"]) == (staged_mb, staged_mb)
    assert stats["total_host_mb"] == host_mb
    assert list(stats) == [
        "num_users",
        "total_tokens",
        "total_gpu_mb",
        "gpu_staging_mb",
        "gpu_chunk_map_mb",
        "chunk_map_bytes",
        "chunk_map_device",
        "total_host_mb",
    ]


def test_steady_state_staging_footprint_is_every_slot_holding_an_average_transfer(
    fake_torch: Any,
) -> None:
    store = store_with(fake_torch, {"u0": 2, "u1": 6}, num_staging_slots=3)
    assert store.get_bench_summary()["steady_state_staging_mb"] == 0.0

    for user_id in ("u0", "u1"):
        store.prefetch_user_to_gpu(user_id)
        store.release_staging(user_id)

    summary = store.get_bench_summary()
    average_transfer_bytes = (payload_bytes(2) + payload_bytes(6)) / 2
    assert summary["total_transfers"] == 2
    assert summary["steady_state_staging_mb"] == 3 * average_transfer_bytes / MB
    assert list(summary)[-1] == "steady_state_staging_mb"
    assert store.get_stats()["gpu_staging_mb"] == 0.0
