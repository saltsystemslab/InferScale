from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .chunk_map import GPUChunkMap

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UserMemory:
    kv_by_layer: dict[str, Any]
    num_tokens: int
    token_ids: list[int] | None = None


class DeviceChunkSelectionError(RuntimeError):
    """A device result cannot be mapped safely to corpus chunk rows."""


class GPUMemoryStore:
    """Thread-safe process-local store for composed GPU KV tensors."""

    def __init__(self, device: str = "cuda:0") -> None:
        self._device = device
        self._memories: dict[str, UserMemory] = {}
        self._bytes_by_user: dict[str, int] = {}
        self._total_tokens = 0
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._chunk_lookup: GPUChunkMap | None = None
        self._chunk_memories: tuple[tuple[str, UserMemory], ...] = ()
        self._device_row_maps: dict[int, Any] = {}

    def add_user_memory(
        self,
        user_id: str,
        kv_by_layer: dict[str, Any],
        num_tokens: int,
        token_ids: list[int] | None = None,
    ) -> None:
        device_kv = {
            layer_name: _contiguous_on_device(tensor, self._device)
            for layer_name, tensor in kv_by_layer.items()
        }
        byte_count = kv_nbytes(device_kv)
        with self._lock:
            self._remove_locked(user_id)
            self._invalidate_chunk_lookup_locked()
            self._memories[user_id] = UserMemory(
                kv_by_layer=device_kv,
                num_tokens=num_tokens,
                token_ids=list(token_ids) if token_ids is not None else None,
            )
            self._bytes_by_user[user_id] = byte_count
            self._total_tokens += num_tokens
            self._total_bytes += byte_count
        logger.info(
            "Stored memory for user %s: %d tokens, %d layers",
            user_id,
            num_tokens,
            len(device_kv),
        )

    def get_user_memory(self, user_id: str) -> UserMemory | None:
        with self._lock:
            return self._memories.get(user_id)

    def peek_user_memory(self, user_id: str) -> UserMemory | None:
        """Metadata read; identical to get_user_memory for the GPU store."""
        return self.get_user_memory(user_id)

    def finalize_chunk_lookup(self) -> None:
        """Build the mandatory GPU ID map for corpus access."""
        with self._lock:
            self._finalize_chunk_lookup_locked()

    def _finalize_chunk_lookup_locked(self) -> None:
        if self._chunk_lookup is None:
            memories = tuple(self._memories.items())
            self._chunk_lookup = GPUChunkMap(
                [chunk_id for chunk_id, _ in memories], device=self._device
            )
            self._chunk_memories = memories

    def get_chunk_memories(
        self, chunk_ids: Sequence[str]
    ) -> list[tuple[str, UserMemory]]:
        """Resolve corpus IDs on GPU, then access payloads by the returned rows."""
        with self._lock:
            self._finalize_chunk_lookup_locked()
            rows = self._chunk_lookup.lookup(chunk_ids).detach().cpu().tolist()
            return [self._chunk_memories[row] for row in rows]

    def build_device_row_map(self, stable_id_items: Iterable[tuple[int, str]]) -> Any:
        with self._lock:
            self._finalize_chunk_lookup_locked()
            return self._bind_device_rows_locked(stable_id_items)

    def _bind_device_rows_locked(self, stable_id_items: Iterable[tuple[int, str]]) -> Any:
        rows = self._chunk_lookup.stable_id_row_map(stable_id_items)
        self._device_row_maps[id(rows)] = weakref.ref(rows)
        return rows

    def _device_rows_locked(self, stable_ids: Any, id_to_row: Any, *, reverse: bool) -> Any:
        validate_row_map_binding(self._device_row_maps, id_to_row)
        return device_chunk_rows(
            stable_ids, id_to_row, device=self._chunk_lookup.device,
            num_rows=len(self._chunk_memories), reverse=reverse,
        )

    def get_device_chunk_memories(
        self, stable_ids: Any, id_to_row: Any, *, reverse: bool = False
    ) -> list[tuple[str, UserMemory]]:
        with self._lock:
            self._finalize_chunk_lookup_locked()
            rows = self._device_rows_locked(
                stable_ids, id_to_row, reverse=reverse
            ).detach().cpu().tolist()
            return [self._chunk_memories[row] for row in rows]

    def _invalidate_chunk_lookup_locked(self) -> None:
        if self._chunk_lookup is not None:
            self._chunk_lookup.close()
            self._chunk_lookup = None
        self._chunk_memories = ()
        self._device_row_maps.clear()

    def close_chunk_lookup(self) -> None:
        with self._lock:
            self._invalidate_chunk_lookup_locked()

    def remove_user_memory(self, user_id: str) -> bool:
        with self._lock:
            removed = self._remove_locked(user_id)
        if removed:
            logger.info("Removed memory for user %s", user_id)
        return removed

    def get_all_user_ids(self) -> list[str]:
        with self._lock:
            return list(self._memories)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            lookup_bytes = self._chunk_lookup.nbytes if self._chunk_lookup else 0
            stats = {
                "num_users": len(self._memories),
                "total_tokens": self._total_tokens,
                "total_gpu_mb": bytes_to_mb(self._total_bytes + lookup_bytes),
            }
            if self._chunk_lookup is not None:
                stats.update(
                    gpu_chunk_map_mb=bytes_to_mb(lookup_bytes),
                    chunk_map_bytes=lookup_bytes,
                    chunk_map_device=str(self._chunk_lookup.device),
                )
            return stats

    # ── Staging/metrics protocol ──────────────────────────────────
    # Concrete no-op defaults: memories are already GPU-resident, so there is
    # nothing to stage, and no transfers to meter. CpuPinnedMemoryStore
    # overrides every member; callers use plain calls, never getattr probes.

    num_staging_slots = 0

    def prefetch_user_to_gpu(self, user_id: str) -> bool:
        del user_id
        return False

    def release_staging(self, user_id: str) -> None:
        del user_id

    def get_bench_summary(self) -> dict[str, float | int]:
        return {}

    def last_transfer_record(self) -> Any:
        return None

    def transfer_count(self) -> int:
        return 0

    def transfer_totals(self) -> dict[str, float]:
        return {}

    def reset_bench_metrics(self) -> None:
        return

    def _remove_locked(self, user_id: str) -> bool:
        memory = self._memories.pop(user_id, None)
        if memory is None:
            return False
        self._invalidate_chunk_lookup_locked()
        self._total_tokens -= memory.num_tokens
        self._total_bytes -= self._bytes_by_user.pop(user_id, 0)
        del memory.kv_by_layer
        return True


def validate_row_map_binding(bindings: dict[int, Any], id_to_row: Any) -> None:
    binding = bindings.get(id(id_to_row))
    if binding is None or binding() is not id_to_row:
        raise DeviceChunkSelectionError("The KV row map is not bound to the current corpus.")


def device_chunk_rows(
    stable_ids: Any, id_to_row: Any, *, device: Any, num_rows: int, reverse: bool
) -> Any:
    """Resolve a Jasper result using CUDA tensors, rejecting unsafe selections."""
    import torch

    expected_device = torch.device(device)
    if expected_device.type != "cuda":
        raise DeviceChunkSelectionError("The KV chunk row map must reside on CUDA.")
    for name, tensor in (("Jasper stable IDs", stable_ids), ("KV row map", id_to_row)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or tensor.numel() < 1:
            raise DeviceChunkSelectionError(f"{name} must be a nonempty one-dimensional tensor.")
        if tensor.device.type != "cuda" or tensor.device != expected_device:
            raise DeviceChunkSelectionError(f"{name} must reside on the chunk map's CUDA device.")
        if tensor.dtype not in (torch.int32, torch.int64):
            raise DeviceChunkSelectionError(f"{name} must contain integer IDs or rows.")
    stable_ids = stable_ids.to(dtype=torch.long)
    valid_range = (stable_ids >= 0) & (stable_ids < id_to_row.numel())
    safe_ids = torch.where(valid_range, stable_ids, torch.zeros_like(stable_ids))
    rows = id_to_row.index_select(0, safe_ids).to(dtype=torch.long)
    valid = (valid_range & (rows >= 0) & (rows < num_rows)).all()
    if stable_ids.numel() > 1:
        sorted_ids = torch.sort(stable_ids).values
        valid = valid & (sorted_ids[1:] != sorted_ids[:-1]).all()
    if not bool(valid.item()):
        raise DeviceChunkSelectionError(
            "Jasper returned a padded, invalid, unmapped, or duplicate stable ID."
        )
    return torch.flip(rows, dims=(0,)) if reverse else rows


def _contiguous_on_device(tensor: Any, device: str) -> Any:
    current_device = getattr(tensor, "device", None)
    if current_device is not None and str(current_device) != device:
        try:
            tensor = tensor.to(device=device)
        except TypeError:
            tensor = tensor.to(device)
    contiguous = getattr(tensor, "contiguous", None)
    if callable(contiguous):
        return contiguous()
    return tensor


def bytes_to_mb(byte_count: int | float) -> float:
    return byte_count / (1024 * 1024)


def kv_nbytes(kv_by_layer: dict[str, Any]) -> int:
    return sum(tensor_nbytes(tensor) for tensor in kv_by_layer.values())


def tensor_nbytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    element_size = getattr(tensor, "element_size", None)
    nelement = getattr(tensor, "nelement", None)
    if callable(element_size) and callable(nelement):
        return int(element_size() * nelement())
    return 0
