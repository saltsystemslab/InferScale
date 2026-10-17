"""Pinned-host KV memory store streamed to the GPU over PCIe.

The authoritative pre-encoded KV lives in page-locked (pinned) host memory.
A small pool of GPU staging buffers receives async H2D copies on a dedicated
CUDA copy stream; per-layer CUDA events let the connector's per-layer scatter
loop start on layer k as soon as layer k's bytes have landed, overlapping
the remaining transfers with compute.

Three correctness details matter: staging tensors are registered on the
consuming stream via ``record_stream`` (the caching allocator must not reuse
them while scatter kernels are in flight), the layer view is lazy (no
all-layer sync from ``values()`` or ``items()``), and the compute stream is
resolved at use time instead of captured at construction.

torch is imported inside the class so this module stays importable on
machines without torch.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..units import bytes_to_mb
from .chunk_lookup import ChunkLookup
from .connector_utils import DEFAULT_KV_STAGING_SLOTS
from .memory_store import UserMemory, kv_nbytes
from .staging import LayerKVView, StagingSlot
from .transfer_metrics import TransferMetrics, TransferRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HostUserMemory:
    """Pinned-host storage for one memory's pre-encoded KV."""

    kv_by_layer_host: dict[str, Any]
    num_tokens: int
    token_ids: list[int] | None
    nbytes: int


class CpuPinnedMemoryStore:
    """GPUMemoryStore-compatible store backed by pinned host memory.

    ``get_user_memory`` hands out a ``UserMemory`` whose ``kv_by_layer`` is a
    ``LayerKVView`` over GPU staging buffers. Call ``prefetch_user_to_gpu``
    before the per-layer loop for overlap and ``release_staging`` after the
    request so the slot returns to the pool and the transfer is recorded.
    """

    def __init__(
        self, device: str = "cuda:0", num_staging_slots: int = DEFAULT_KV_STAGING_SLOTS
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CpuPinnedMemoryStore requires a CUDA device.")
        if num_staging_slots < 1:
            raise ValueError("num_staging_slots must be >= 1.")

        self._torch = torch
        self._device = torch.device(device)
        with torch.cuda.device(self._device):
            self._copy_stream = torch.cuda.Stream(device=self._device)
        self._host: dict[str, HostUserMemory] = {}
        self._chunk_lookup: ChunkLookup[HostUserMemory] | None = None
        self._slots: dict[str, StagingSlot] = {}
        # The connector sizes its sliding prefetch window to this.
        self.num_staging_slots = num_staging_slots
        self._lock = threading.Lock()
        self._metrics = TransferMetrics()

    # Write path

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

        # Release any staging of a previous registration before swapping the
        # host payload, so its transfer record keeps the old sizes.
        self.release_staging(user_id)
        with self._lock:
            self._close_lookup_locked()
            self._host[user_id] = HostUserMemory(
                kv_by_layer_host=pinned,
                num_tokens=num_tokens,
                token_ids=list(token_ids) if token_ids is not None else None,
                nbytes=nbytes,
            )
        logger.debug(
            "Stored pinned-host memory for user %s: %d tokens, %d layers, %.1f MB",
            user_id,
            num_tokens,
            len(pinned),
            bytes_to_mb(nbytes),
        )

    def remove_user_memory(self, user_id: str) -> bool:
        with self._lock:
            present = user_id in self._host
        if not present:
            return False
        # Release while the host entry still exists so a still-staged memory's
        # transfer record carries its real token and byte sizes.
        self.release_staging(user_id)
        with self._lock:
            removed = self._host.pop(user_id, None) is not None
            if removed:
                self._close_lookup_locked()
        if removed:
            logger.debug("Removed pinned-host memory for user %s", user_id)
        return removed

    def get_all_user_ids(self) -> list[str]:
        with self._lock:
            return list(self._host)

    # Read path

    def peek_user_memory(self, user_id: str) -> UserMemory | None:
        """Metadata-only read (token ids and count) that stages nothing.

        Prefix matching on the scheduler side must use this; get_user_memory
        is the load-path accessor and stages the full payload.
        """
        with self._lock:
            host_memory = self._host.get(user_id)
        if host_memory is None:
            return None
        return UserMemory(
            kv_by_layer={}, num_tokens=host_memory.num_tokens, token_ids=host_memory.token_ids
        )

    def get_user_memory(self, user_id: str) -> UserMemory | None:
        with self._lock:
            host_memory = self._host.get(user_id)
        if host_memory is None:
            return None
        return self._staged_memory(user_id, host_memory)

    def prefetch_user_to_gpu(self, user_id: str) -> bool:
        """Issue async H2D copies for every layer on the copy stream.

        Returns True if the memory is staged afterwards (including when it
        already was) and False if the user has no stored memory.
        """
        with self._lock:
            host_memory = self._host.get(user_id)
        if host_memory is None:
            return False
        return self._prefetch_host_memory(user_id, host_memory)

    def release_staging(self, user_id: str) -> None:
        """Free the user's staging slot and record the transfer's metrics."""
        with self._lock:
            slot = self._slots.pop(user_id, None)
            host_memory = self._host.get(user_id)
        if slot is None:
            return
        slot.end_event.synchronize()
        self._metrics.add(
            TransferRecord(
                user_id=user_id,
                num_tokens=host_memory.num_tokens if host_memory is not None else 0,
                num_bytes=host_memory.nbytes if host_memory is not None else 0,
                h2d_latency_ms=float(slot.start_event.elapsed_time(slot.end_event)),
                staging_stall_ms=slot.view.stall_ms,
            )
        )

    # Corpus access through the GPU lookup

    def finalize_packed(self) -> None:
        """Payloads stay as per-chunk pinned buffers; nothing to pack."""
        return

    def finalize_chunk_lookup(self) -> None:
        with self._lock:
            self._lookup_locked()

    def close_chunk_lookup(self) -> None:
        with self._lock:
            self._close_lookup_locked()

    def get_chunk_memories(self, chunk_ids: Sequence[str]) -> list[tuple[str, UserMemory]]:
        with self._lock:
            resolved = self._lookup_locked().resolve(chunk_ids)
        return [(chunk_id, self._staged_memory(chunk_id, memory)) for chunk_id, memory in resolved]

    def build_device_row_map(self, stable_id_items: Iterable[tuple[int, str]]) -> Any:
        with self._lock:
            return self._lookup_locked().bind_stable_ids(stable_id_items)

    def get_device_chunk_memories(
        self, stable_ids: Any, id_to_row: Any, *, reverse: bool = False
    ) -> list[tuple[str, UserMemory]]:
        with self._lock:
            resolved = self._lookup_locked().resolve_device(stable_ids, id_to_row, reverse=reverse)
        return [(chunk_id, self._staged_memory(chunk_id, memory)) for chunk_id, memory in resolved]

    # Statistics

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            lookup = self._chunk_lookup
            lookup_bytes = lookup.nbytes if lookup is not None else 0
            staging_mb = bytes_to_mb(
                sum(kv_nbytes(slot.gpu_tensors) for slot in self._slots.values())
            )
            return {
                "num_users": len(self._host),
                "total_tokens": sum(memory.num_tokens for memory in self._host.values()),
                # total_gpu_mb keeps the GPU-store meaning of resident HBM, which
                # for this store is the staging pool rather than the payload.
                "total_gpu_mb": staging_mb + bytes_to_mb(lookup_bytes),
                "gpu_staging_mb": staging_mb,
                "gpu_chunk_map_mb": bytes_to_mb(lookup_bytes),
                "chunk_map_bytes": lookup_bytes,
                "chunk_map_device": str(lookup.device) if lookup is not None else None,
                "total_host_mb": bytes_to_mb(sum(memory.nbytes for memory in self._host.values())),
            }

    def get_bench_summary(self) -> dict[str, float | int]:
        summary = self._metrics.summary()
        # The store sizes its own staging pool, so it reports the steady-state
        # HBM footprint (slots times the average transfer) itself; outside an
        # injection step the instantaneous staging footprint is zero.
        transfers = int(summary["total_transfers"])
        avg_transfer_bytes = summary["total_bytes_transferred"] / transfers if transfers else 0.0
        summary["steady_state_staging_mb"] = bytes_to_mb(
            self.num_staging_slots * avg_transfer_bytes
        )
        return summary

    def transfer_totals(self) -> dict[str, float]:
        return self._metrics.totals()

    # Internals

    def _staged_memory(self, user_id: str, host_memory: HostUserMemory) -> UserMemory:
        self._prefetch_host_memory(user_id, host_memory)
        with self._lock:
            slot = self._slots.get(user_id)
        if slot is None:
            raise RuntimeError(
                f"Staging slot for user {user_id} was evicted before use; "
                "increase the staging slot count."
            )
        return UserMemory(
            kv_by_layer=slot.view,  # type: ignore[arg-type]  # dict-like lazy view
            num_tokens=host_memory.num_tokens,
            token_ids=host_memory.token_ids,
        )

    def _prefetch_host_memory(self, user_id: str, host_memory: HostUserMemory) -> bool:
        """Stage an already resolved payload without consulting the ID map."""
        torch = self._torch
        with self._lock:
            if user_id in self._slots:
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

        view = LayerKVView(
            gpu_tensors=staging,
            layer_events=layer_events,
            compute_stream_fn=self._current_compute_stream,
        )
        with self._lock:
            self._slots[user_id] = StagingSlot(
                gpu_tensors=staging, start_event=start_event, end_event=end_event, view=view
            )
        self._evict_over_capacity(keep=user_id)
        return True

    def _evict_over_capacity(self, keep: str) -> None:
        with self._lock:
            if len(self._slots) <= self.num_staging_slots:
                return
            victim = next((candidate for candidate in self._slots if candidate != keep), None)
        if victim is None:
            return
        # The connector's sliding prefetch window keeps residency at or below
        # the pool size, so eviction indicates an undersized pool.
        logger.warning(
            "Evicting staging slot for user %s; consider raising num_staging_slots", victim
        )
        self.release_staging(victim)

    def _current_compute_stream(self) -> Any:
        return self._torch.cuda.current_stream(self._device)

    def _lookup_locked(self) -> ChunkLookup[HostUserMemory]:
        if self._chunk_lookup is None:
            self._chunk_lookup = ChunkLookup(self._host.items(), device=str(self._device))
        return self._chunk_lookup

    def _close_lookup_locked(self) -> None:
        if self._chunk_lookup is not None:
            self._chunk_lookup.close()
            self._chunk_lookup = None
