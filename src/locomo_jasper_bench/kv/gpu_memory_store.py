from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UserMemory:
    kv_by_layer: dict[str, Any]
    num_tokens: int
    token_ids: list[int] | None = None


class GPUMemoryStore:
    """Thread-safe process-local store for benchmark-composed GPU KV tensors."""

    def __init__(self, device: str = "cuda:0") -> None:
        self._device = device
        self._memories: dict[str, UserMemory] = {}
        self._bytes_by_user: dict[str, int] = {}
        self._total_tokens = 0
        self._total_bytes = 0
        self._lock = threading.Lock()

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

    def remove_user_memory(self, user_id: str) -> bool:
        with self._lock:
            removed = self._remove_locked(user_id)
        if removed:
            logger.info("Removed memory for user %s", user_id)
        return removed

    def get_all_user_ids(self) -> list[str]:
        with self._lock:
            return list(self._memories)

    def get_stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "num_users": len(self._memories),
                "total_tokens": self._total_tokens,
                "total_gpu_mb": bytes_to_mb(self._total_bytes),
            }

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
        self._total_tokens -= memory.num_tokens
        self._total_bytes -= self._bytes_by_user.pop(user_id, 0)
        del memory.kv_by_layer
        return True


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
    return sum(_tensor_nbytes(tensor) for tensor in kv_by_layer.values())


def _tensor_nbytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    element_size = getattr(tensor, "element_size", None)
    nelement = getattr(tensor, "nelement", None)
    if callable(element_size) and callable(nelement):
        return int(element_size() * nelement())
    return 0
