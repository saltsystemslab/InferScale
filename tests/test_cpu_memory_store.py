from __future__ import annotations

from types import SimpleNamespace

import pytest

from locomo_jasper_bench.kv.cpu_memory_store import (
    TransferRecord,
    _BenchMetrics,
    _LayerKVView,
    overlap_ratio,
)
from locomo_jasper_bench.kv import gpu_registry
from locomo_jasper_bench.kv.gpu_memory_store import GPUMemoryStore


def _record(
    *,
    user_id: str = "u",
    h2d: float = 10.0,
    stall: float = 0.0,
    num_bytes: int = 1000,
) -> TransferRecord:
    return TransferRecord(
        user_id=user_id,
        num_tokens=16,
        num_bytes=num_bytes,
        h2d_latency_ms=h2d,
        staging_stall_ms=stall,
    )


class TestBenchMetrics:
    def test_empty_summary_shape(self) -> None:
        summary = _BenchMetrics().summary()
        assert summary["total_transfers"] == 0
        assert summary["total_bytes_transferred"] == 0
        assert summary["avg_h2d_latency_ms"] == 0.0
        assert summary["total_staging_stall_ms"] == 0.0

    def test_totals_and_averages(self) -> None:
        metrics = _BenchMetrics()
        metrics.add(_record(h2d=10.0, stall=5.0, num_bytes=1_000))
        metrics.add(_record(h2d=20.0, stall=0.0, num_bytes=3_000))
        summary = metrics.summary()
        assert summary["total_transfers"] == 2
        assert summary["total_bytes_transferred"] == 4_000
        assert summary["avg_h2d_latency_ms"] == pytest.approx(15.0)
        # overlaps: 1 - 5/10 = 0.5 and 1 - 0/20 = 1.0
        assert summary["avg_overlap_ratio"] == pytest.approx(0.75)
        assert summary["total_staging_stall_ms"] == pytest.approx(5.0)
        assert summary["avg_staging_stall_ms"] == pytest.approx(2.5)

    def test_overlap_ratio_clamped_to_unit_interval(self) -> None:
        # A stall longer than the transfer itself must clamp to 0.
        assert overlap_ratio(_record(h2d=10.0, stall=50.0)) == 0.0
        assert overlap_ratio(_record(h2d=0.0, stall=0.0)) == 0.0
        metrics = _BenchMetrics()
        metrics.add(_record(h2d=10.0, stall=50.0))
        assert metrics.summary()["avg_overlap_ratio"] == 0.0

    def test_percentiles_and_last_record(self) -> None:
        metrics = _BenchMetrics()
        for latency in (1.0, 2.0, 3.0, 4.0, 100.0):
            metrics.add(_record(h2d=latency))
        summary = metrics.summary()
        assert summary["p50_h2d_latency_ms"] == 3.0
        # Shared results._percentile semantics: linear interpolation, so
        # p95 of five values sits between the 4th and 5th.
        assert summary["p95_h2d_latency_ms"] == pytest.approx(4 + 0.8 * (100 - 4))
        assert metrics.last_record().h2d_latency_ms == 100.0
        assert _BenchMetrics().last_record() is None


class _FakeEvent:
    def __init__(self, ready: bool) -> None:
        self._ready = ready
        self.synchronized = 0
        self.waited_by: list[object] = []

    def query(self) -> bool:
        return self._ready

    def synchronize(self) -> None:
        self._ready = True
        self.synchronized += 1


class _FakeStream:
    def __init__(self) -> None:
        self.waited_events: list[object] = []

    def wait_event(self, event: object) -> None:
        self.waited_events.append(event)


class _FakeTensor(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(recorded_streams=[])

    def record_stream(self, stream: object) -> None:
        self.recorded_streams.append(stream)


def _view(ready: bool = True) -> tuple[_LayerKVView, dict, dict, _FakeStream]:
    tensors = {"layer.0": _FakeTensor(), "layer.1": _FakeTensor()}
    events = {"layer.0": _FakeEvent(ready), "layer.1": _FakeEvent(ready)}
    stream = _FakeStream()
    view = _LayerKVView(
        gpu_tensors=tensors,
        layer_events=events,
        compute_stream_fn=lambda: stream,
    )
    return view, tensors, events, stream


class TestLayerKVView:
    def test_getitem_waits_once_per_layer_and_records_stream(self) -> None:
        view, tensors, events, stream = _view(ready=True)
        first = view["layer.0"]
        again = view["layer.0"]
        assert first is again is tensors["layer.0"]
        # Ready event: no blocking synchronize, but the compute stream still
        # orders after the copy and the tensor is registered on it.
        assert events["layer.0"].synchronized == 0
        assert stream.waited_events == [events["layer.0"]]
        assert tensors["layer.0"].recorded_streams == [stream]
        assert view.stall_ms == 0.0

    def test_getitem_accumulates_stall_when_copy_pending(self) -> None:
        view, _, events, _ = _view(ready=False)
        view["layer.0"]
        assert events["layer.0"].synchronized == 1
        assert view.stall_ms >= 0.0
        # Second layer stalls independently.
        view["layer.1"]
        assert events["layer.1"].synchronized == 1

    def test_get_semantics(self) -> None:
        view, tensors, _, _ = _view()
        assert view.get("layer.0") is tensors["layer.0"]
        assert view.get("missing") is None
        assert view.get("missing", "fallback") == "fallback"

    def test_dict_surface(self) -> None:
        view, tensors, _, _ = _view()
        assert set(view.keys()) == {"layer.0", "layer.1"}
        assert len(view) == 2
        assert "layer.0" in view
        assert set(iter(view)) == {"layer.0", "layer.1"}
        assert list(view.values()) == [tensors["layer.0"], tensors["layer.1"]]
        assert dict(view.items()) == tensors

    def test_values_is_lazy(self) -> None:
        view, _, events, stream = _view(ready=True)
        iterator = view.values()
        # Nothing consumed yet: no event interaction at all.
        assert stream.waited_events == []
        next(iterator)
        assert len(stream.waited_events) == 1


class TestRegistryBackendSelection:
    def setup_method(self) -> None:
        gpu_registry._STORES.clear()
        gpu_registry._BACKENDS.clear()

    teardown_method = setup_method

    def test_default_backend_is_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCOMO_KV_STORE_BACKEND", raising=False)
        store = gpu_registry.get_gpu_memory_store("ns-default")
        assert isinstance(store, GPUMemoryStore)

    def test_env_var_resolves_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created: list[int] = []
        monkeypatch.setenv("LOCOMO_KV_STORE_BACKEND", "cpu-pinned")
        import locomo_jasper_bench.kv.cpu_memory_store as cpu_module

        monkeypatch.setattr(
            cpu_module,
            "CpuPinnedMemoryStore",
            lambda num_staging_slots: created.append(num_staging_slots) or object(),
        )
        gpu_registry.get_gpu_memory_store("ns-env")
        assert created == [4]

    def test_explicit_backend_and_staging_slots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created: list[int] = []
        import locomo_jasper_bench.kv.cpu_memory_store as cpu_module

        monkeypatch.setattr(
            cpu_module,
            "CpuPinnedMemoryStore",
            lambda num_staging_slots: created.append(num_staging_slots) or object(),
        )
        gpu_registry.get_gpu_memory_store("ns-cpu", backend="cpu-pinned", num_staging_slots=8)
        assert created == [8]

    def test_backend_mismatch_raises(self) -> None:
        gpu_registry.get_gpu_memory_store("ns-mismatch", backend="gpu")
        with pytest.raises(RuntimeError, match="already exists with backend 'gpu'"):
            gpu_registry.get_gpu_memory_store("ns-mismatch", backend="cpu-pinned")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown KV store backend"):
            gpu_registry.get_gpu_memory_store("ns-bad", backend="tpu")

    def test_bench_helpers_noop_for_gpu_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCOMO_KV_STORE_BACKEND", raising=False)
        gpu_registry.get_gpu_memory_store("ns-gpu")
        assert gpu_registry.namespace_bench_summary("ns-gpu") == {}
        assert gpu_registry.namespace_last_transfer("ns-gpu") is None
        gpu_registry.reset_namespace_bench_metrics("ns-gpu")  # must not raise

    def test_namespace_stats_empty_default_includes_host_mb(self) -> None:
        stats = gpu_registry.namespace_stats("ns-nonexistent")
        assert stats["total_host_mb"] == 0.0


try:
    import torch
except ImportError:  # torch is absent off the GPU box; pure-python tests above still run
    torch = None

requires_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="CUDA device required"
)


@requires_cuda
class TestCpuPinnedMemoryStoreOnGpu:
    """Round-trip tests for the real store; these run on the GPU box only."""

    def _store(self, num_staging_slots: int = 2):
        from locomo_jasper_bench.kv.cpu_memory_store import CpuPinnedMemoryStore

        return CpuPinnedMemoryStore(num_staging_slots=num_staging_slots)

    def _kv(self, tokens: int = 8) -> dict[str, object]:
        return {
            f"layer.{i}": torch.randn(2, tokens, 2, 4, device="cuda", dtype=torch.float16)
            for i in range(3)
        }

    def test_add_prefetch_get_release_roundtrip(self) -> None:
        store = self._store()
        kv = self._kv()
        store.add_user_memory("u0", kv, num_tokens=8, token_ids=list(range(8)))

        host_stats = store.get_stats()
        assert host_stats["num_users"] == 1
        assert host_stats["total_host_mb"] > 0

        memory = store.get_user_memory("u0")
        assert memory is not None
        for layer_name, source in kv.items():
            staged = memory.kv_by_layer[layer_name]
            assert staged.is_cuda
            assert torch.equal(staged, source)

        store.release_staging("u0")
        summary = store.get_bench_summary()
        assert summary["total_transfers"] == 1
        assert summary["total_bytes_transferred"] > 0

    def test_host_buffers_are_pinned(self) -> None:
        store = self._store()
        store.add_user_memory("u0", self._kv(), num_tokens=8, token_ids=list(range(8)))
        host = store._host["u0"]
        assert all(t.is_pinned() for t in host.kv_by_layer_host.values())

    def test_staging_pool_evicts_lru(self) -> None:
        store = self._store(num_staging_slots=1)
        store.add_user_memory("u0", self._kv(), num_tokens=8, token_ids=list(range(8)))
        store.add_user_memory("u1", self._kv(), num_tokens=8, token_ids=list(range(8)))
        store.prefetch_user_to_gpu("u0")
        store.prefetch_user_to_gpu("u1")
        assert "u0" not in store._staging
        assert "u1" in store._staging

    def test_reset_bench_metrics(self) -> None:
        store = self._store()
        store.add_user_memory("u0", self._kv(), num_tokens=8, token_ids=list(range(8)))
        store.prefetch_user_to_gpu("u0")
        store.release_staging("u0")
        assert store.get_bench_summary()["total_transfers"] == 1
        store.reset_bench_metrics()
        assert store.get_bench_summary()["total_transfers"] == 0

    def test_peek_reads_metadata_without_staging(self) -> None:
        store = self._store()
        store.add_user_memory("u0", self._kv(), num_tokens=8, token_ids=list(range(8)))

        peeked = store.peek_user_memory("u0")

        assert peeked is not None
        assert peeked.num_tokens == 8
        assert peeked.token_ids == list(range(8))
        assert store._staging == {}
        assert store.get_bench_summary()["total_transfers"] == 0
        assert store.peek_user_memory("missing") is None


def test_gpu_store_peek_is_get() -> None:
    store = GPUMemoryStore()
    store.add_user_memory("u0", {"layer.0": object()}, num_tokens=4, token_ids=[1, 2, 3, 4])

    assert store.peek_user_memory("u0") is store.get_user_memory("u0")
    assert store.peek_user_memory("missing") is None
