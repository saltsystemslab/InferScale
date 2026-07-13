"""Pinned-host KV memory store streamed to GPU over PCIe.

Drop-in replacement for GPUMemoryStore where the authoritative pre-encoded KV
lives in page-locked (pinned) host memory. A small pool of GPU staging buffers
receives async H2D copies on a dedicated CUDA copy stream; per-layer CUDA
events let the connector's per-layer scatter loop start on layer k as soon as
layer k's bytes have landed, overlapping the remaining transfers with compute.

Ported from ai-memory-code/memory_connector/cpu_memory_store.py with three
fixes: staging tensors are registered on the consuming stream via
``record_stream`` (the caching allocator must not reuse them while scatter
kernels are in flight), the layer view is lazy (no all-layer sync from
``values()``/``items()``), and the compute stream is resolved at use time
instead of captured at construction.

torch is imported inside the class so this module stays importable (and the
pure-python pieces unit-testable) on machines without torch.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..results import _percentile
from .gpu_memory_store import UserMemory, _kv_nbytes

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _HostUserMemory:
    """Pinned-host storage for one user's pre-encoded KV."""

    kv_by_layer_host: dict[str, Any]
    num_tokens: int
    token_ids: list[int] | None
    nbytes: int


class _LayerKVView:
    """Dict-like view over per-layer GPU staging tensors.

    First access to a layer waits on that layer's H2D copy event (timing any
    blocking wait into ``stall_ms``), makes the compute stream wait on the
    event, and registers the tensor on the compute stream so the allocator
    cannot reuse it until in-flight consumers finish.
    """

    def __init__(
        self,
        gpu_tensors: dict[str, Any],
        layer_events: dict[str, Any],
        compute_stream_fn: Callable[[], Any],
    ) -> None:
        self._gpu = gpu_tensors
        self._events = layer_events
        self._compute_stream_fn = compute_stream_fn
        self._waited: set[str] = set()
        self.stall_ms = 0.0

    def __getitem__(self, key: str) -> Any:
        tensor = self._gpu[key]
        if key not in self._waited:
            event = self._events.get(key)
            compute_stream = self._compute_stream_fn()
            if event is not None:
                query = getattr(event, "query", None)
                if callable(query) and not query():
                    stall_started = time.perf_counter()
                    event.synchronize()
                    self.stall_ms += (time.perf_counter() - stall_started) * 1e3
                if compute_stream is not None:
                    compute_stream.wait_event(event)
            record_stream = getattr(tensor, "record_stream", None)
            if callable(record_stream) and compute_stream is not None:
                record_stream(compute_stream)
            self._waited.add(key)
        return tensor

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._gpu:
            return default
        return self[key]

    def __contains__(self, key: str) -> bool:
        return key in self._gpu

    def __iter__(self) -> Iterator[str]:
        return iter(self._gpu)

    def __len__(self) -> int:
        return len(self._gpu)

    def keys(self):
        return self._gpu.keys()

    def values(self) -> Iterator[Any]:
        return (self[key] for key in self._gpu)

    def items(self) -> Iterator[tuple[str, Any]]:
        return ((key, self[key]) for key in self._gpu)


@dataclass(slots=True)
class TransferRecord:
    """One user's H2D staging transfer.

    staging_stall_ms is the time consumers blocked waiting for layer copies,
    i.e. the transfer's critical-path exposure; the rest of h2d_latency_ms
    overlapped with compute.
    """

    user_id: str
    num_tokens: int
    num_bytes: int
    h2d_latency_ms: float
    staging_stall_ms: float


def overlap_ratio(record: TransferRecord) -> float:
    """Fraction of the transfer hidden behind compute, clamped to [0, 1]."""
    if record.h2d_latency_ms <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - record.staging_stall_ms / record.h2d_latency_ms))


@dataclass
class _BenchMetrics:
    """Transfer metrics for the pinned-host store.

    Records are unbounded so the summary's percentiles and totals cover the
    same population as the lifetime counters; a full run's records cost on
    the order of 100 bytes each.
    """

    records: deque = field(default_factory=deque)
    total_bytes: int = 0
    total_transfers: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, record: TransferRecord) -> None:
        with self.lock:
            self.records.append(record)
            self.total_bytes += record.num_bytes
            self.total_transfers += 1

    def last_record(self) -> TransferRecord | None:
        with self.lock:
            return self.records[-1] if self.records else None

    def summary(self) -> dict[str, float | int]:
        with self.lock:
            if not self.records:
                return {
                    "total_transfers": 0,
                    "total_bytes_transferred": 0,
                    "avg_h2d_latency_ms": 0.0,
                    "p50_h2d_latency_ms": 0.0,
                    "p95_h2d_latency_ms": 0.0,
                    "p99_h2d_latency_ms": 0.0,
                    "avg_overlap_ratio": 0.0,
                    "avg_effective_bandwidth_gb_per_s": 0.0,
                    "total_staging_stall_ms": 0.0,
                    "avg_staging_stall_ms": 0.0,
                    "p95_staging_stall_ms": 0.0,
                }
            latencies = sorted(record.h2d_latency_ms for record in self.records)
            stalls = sorted(record.staging_stall_ms for record in self.records)
            overlaps = [
                overlap_ratio(record)
                for record in self.records
                if record.h2d_latency_ms > 0
            ]
            bandwidths = [
                (record.num_bytes / 1e9) / (record.h2d_latency_ms / 1e3)
                for record in self.records
                if record.h2d_latency_ms > 0
            ]

            return {
                "total_transfers": self.total_transfers,
                "total_bytes_transferred": self.total_bytes,
                "avg_h2d_latency_ms": sum(latencies) / len(latencies),
                "p50_h2d_latency_ms": _percentile(latencies, 0.50),
                "p95_h2d_latency_ms": _percentile(latencies, 0.95),
                "p99_h2d_latency_ms": _percentile(latencies, 0.99),
                "avg_overlap_ratio": sum(overlaps) / len(overlaps) if overlaps else 0.0,
                "avg_effective_bandwidth_gb_per_s": (
                    sum(bandwidths) / len(bandwidths) if bandwidths else 0.0
                ),
                "total_staging_stall_ms": sum(stalls),
                "avg_staging_stall_ms": sum(stalls) / len(stalls),
                "p95_staging_stall_ms": _percentile(stalls, 0.95),
            }


class CpuPinnedMemoryStore:
    """GPUMemoryStore-compatible store backed by pinned host memory.

    ``get_user_memory`` hands out ``UserMemory`` whose ``kv_by_layer`` is a
    ``_LayerKVView`` over GPU staging buffers; call ``prefetch_user_to_gpu``
    before the per-layer loop for overlap and ``release_staging`` after the
    request so the slot returns to the pool and metrics get recorded.
    """

    def __init__(self, device: str = "cuda:0", num_staging_slots: int = 4) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CpuPinnedMemoryStore requires a CUDA device.")
        if num_staging_slots < 1:
            raise ValueError("num_staging_slots must be >= 1.")

        self._torch = torch
        self._device = torch.device(device)
        with torch.cuda.device(self._device):
            self._copy_stream = torch.cuda.Stream(device=self._device)

        self._host: dict[str, _HostUserMemory] = {}
        self._staging: dict[str, dict[str, Any]] = {}
        self._layer_events: dict[str, dict[str, Any]] = {}
        self._xfer_events: dict[str, tuple[Any, Any]] = {}
        self._views: dict[str, _LayerKVView] = {}
        # Public: the connector sizes its sliding prefetch window to this.
        self.num_staging_slots = num_staging_slots
        self._lock = threading.Lock()

        self._metrics = _BenchMetrics()

    # ── Write path ────────────────────────────────────────────────

    def add_user_memory(
        self,
        user_id: str,
        kv_by_layer: dict[str, Any],
        num_tokens: int,
        token_ids: list[int] | None = None,
    ) -> None:
        torch = self._torch
        pinned: dict[str, Any] = {}
        nbytes = 0
        for layer_name, tensor in kv_by_layer.items():
            host_buf = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            # Synchronous D2H: this is the registration path, not serving.
            host_buf.copy_(tensor.detach(), non_blocking=False)
            pinned[layer_name] = host_buf
            nbytes += int(host_buf.nbytes)

        # Release any staging for a previous registration of this user before
        # swapping the host payload, so its metrics record the old sizes.
        self.release_staging(user_id)
        with self._lock:
            self._host[user_id] = _HostUserMemory(
                kv_by_layer_host=pinned,
                num_tokens=num_tokens,
                token_ids=list(token_ids) if token_ids is not None else None,
                nbytes=nbytes,
            )
        logger.info(
            "Stored pinned-host memory for user %s: %d tokens, %d layers, %.1f MB",
            user_id,
            num_tokens,
            len(pinned),
            nbytes / (1024 * 1024),
        )

    def peek_user_memory(self, user_id: str) -> UserMemory | None:
        """Metadata-only read (token_ids/num_tokens) that stages nothing.

        Prefix matching on the scheduler side must use this; get_user_memory
        is the load-path accessor and stages the full payload.
        """
        with self._lock:
            host_memory = self._host.get(user_id)
        if host_memory is None:
            return None
        return UserMemory(
            kv_by_layer={},
            num_tokens=host_memory.num_tokens,
            token_ids=host_memory.token_ids,
        )

    def get_user_memory(self, user_id: str) -> UserMemory | None:
        with self._lock:
            host_memory = self._host.get(user_id)
        if host_memory is None:
            return None

        self.prefetch_user_to_gpu(user_id)
        with self._lock:
            view = self._views.get(user_id)
        if view is None:
            raise RuntimeError(
                f"Staging slot for user {user_id} was evicted before use; "
                "increase --kv-staging-slots."
            )
        return UserMemory(
            kv_by_layer=view,  # type: ignore[arg-type]  # duck-typed dict view
            num_tokens=host_memory.num_tokens,
            token_ids=host_memory.token_ids,
        )

    def remove_user_memory(self, user_id: str) -> bool:
        with self._lock:
            present = user_id in self._host
        if not present:
            return False
        # Release while the host entry still exists so a still-staged user's
        # TransferRecord carries its real token/byte sizes, not zeros.
        self.release_staging(user_id)
        with self._lock:
            removed = self._host.pop(user_id, None) is not None
        if removed:
            logger.info("Removed pinned-host memory for user %s", user_id)
        return removed

    def get_all_user_ids(self) -> list[str]:
        with self._lock:
            return list(self._host)

    # ── Read path ─────────────────────────────────────────────────

    def prefetch_user_to_gpu(self, user_id: str) -> bool:
        """Issue async H2D copies for every layer on the copy stream.

        No-op (returns True) if the user is already staged; returns False if
        the user has no stored memory.
        """
        torch = self._torch
        with self._lock:
            host_memory = self._host.get(user_id)
            if host_memory is None:
                return False
            if user_id in self._staging:
                return True

        staging: dict[str, Any] = {}
        layer_events: dict[str, Any] = {}
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(self._copy_stream):
            start_event.record(self._copy_stream)
            for layer_name, host_tensor in host_memory.kv_by_layer_host.items():
                gpu_buf = torch.empty(
                    host_tensor.shape, dtype=host_tensor.dtype, device=self._device
                )
                gpu_buf.copy_(host_tensor, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self._copy_stream)
                staging[layer_name] = gpu_buf
                layer_events[layer_name] = event
            end_event.record(self._copy_stream)

        with self._lock:
            self._staging[user_id] = staging
            self._layer_events[user_id] = layer_events
            self._xfer_events[user_id] = (start_event, end_event)
            self._views[user_id] = _LayerKVView(
                gpu_tensors=staging,
                layer_events=layer_events,
                compute_stream_fn=self._current_compute_stream,
            )

        self._maybe_evict_staging(exclude=user_id)
        return True

    def release_staging(self, user_id: str) -> None:
        """Free the user's staging slot and record the transfer's metrics."""
        with self._lock:
            staging = self._staging.pop(user_id, None)
            self._layer_events.pop(user_id, None)
            xfer_events = self._xfer_events.pop(user_id, None)
            view = self._views.pop(user_id, None)
            host_memory = self._host.get(user_id)

        if staging is None:
            return

        h2d_latency_ms = 0.0
        if xfer_events is not None:
            start_event, end_event = xfer_events
            end_event.synchronize()
            h2d_latency_ms = float(start_event.elapsed_time(end_event))

        self._metrics.add(
            TransferRecord(
                user_id=user_id,
                num_tokens=host_memory.num_tokens if host_memory is not None else 0,
                num_bytes=host_memory.nbytes if host_memory is not None else 0,
                h2d_latency_ms=h2d_latency_ms,
                staging_stall_ms=view.stall_ms if view is not None else 0.0,
            )
        )

    def _maybe_evict_staging(self, exclude: str) -> None:
        with self._lock:
            if len(self._staging) <= self.num_staging_slots:
                return
            victim = next(
                (candidate for candidate in self._staging if candidate != exclude), None
            )
        if victim is None:
            return
        # The connector's sliding prefetch window keeps residency at or below
        # the pool size, so eviction indicates an undersized pool or a bug.
        logger.warning("Evicting staging slot for user %s; consider raising num_staging_slots", victim)
        self.release_staging(victim)

    def _current_compute_stream(self) -> Any:
        return self._torch.cuda.current_stream(self._device)

    # ── Benchmark plumbing ────────────────────────────────────────

    def get_stats(self) -> dict[str, int | float]:
        with self._lock:
            staging_bytes = sum(
                _kv_nbytes(slot) for slot in self._staging.values()
            )
            return {
                "num_users": len(self._host),
                "total_tokens": sum(memory.num_tokens for memory in self._host.values()),
                # total_gpu_mb keeps the GPU-store meaning of "resident HBM":
                # for this store that is the staging pool, not the payload.
                "total_gpu_mb": staging_bytes / (1024 * 1024),
                "gpu_staging_mb": staging_bytes / (1024 * 1024),
                "total_host_mb": sum(memory.nbytes for memory in self._host.values())
                / (1024 * 1024),
            }

    def get_bench_summary(self) -> dict[str, float | int]:
        return self._metrics.summary()

    def last_transfer_record(self) -> TransferRecord | None:
        return self._metrics.last_record()

    def transfer_count(self) -> int:
        return self._metrics.total_transfers

    def reset_bench_metrics(self) -> None:
        self._metrics = _BenchMetrics()
