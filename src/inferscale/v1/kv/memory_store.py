"""In-process GPU KV store for composed memories and the GPU-resident corpus.

Each entry is a ``UserMemory``: per-layer KV tensors plus the token ids they
encode. The connector registers composed request memories here and reads
them back during injection. The ``gpu`` chunk store keeps the corpus here as
well, resolved through a ChunkLookup so serving never looks chunks up by
string on the host.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..units import bytes_to_mb
from .chunk_lookup import ChunkLookup

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UserMemory:
    """Per-layer KV tensors of one registered memory and the tokens they encode."""

    kv_by_layer: dict[str, Any]
    num_tokens: int
    token_ids: list[int] | None = None


class GPUMemoryStore:
    """Thread-safe process-local store for GPU-resident KV tensors."""

    # Entries are already on the GPU: there is nothing to stage and no
    # transfers to meter. CpuPinnedMemoryStore overrides these members.
    num_staging_slots = 0

    def __init__(self, device: str = "cuda:0") -> None:
        self._device = device
        self._memories: dict[str, UserMemory] = {}
        self._bytes_by_user: dict[str, int] = {}
        self._total_tokens = 0
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._chunk_lookup: ChunkLookup[UserMemory] | None = None

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
            self._close_lookup_locked()
            self._memories[user_id] = UserMemory(
                kv_by_layer=device_kv,
                num_tokens=num_tokens,
                token_ids=list(token_ids) if token_ids is not None else None,
            )
            self._bytes_by_user[user_id] = byte_count
            self._total_tokens += num_tokens
            self._total_bytes += byte_count
        logger.debug(
            "Stored memory for user %s: %d tokens, %d layers", user_id, num_tokens, len(device_kv)
        )

    def get_user_memory(self, user_id: str) -> UserMemory | None:
        with self._lock:
            return self._memories.get(user_id)

    def peek_user_memory(self, user_id: str) -> UserMemory | None:
        """Metadata read; identical to get_user_memory for GPU-resident entries."""
        return self.get_user_memory(user_id)

    def remove_user_memory(self, user_id: str) -> bool:
        with self._lock:
            removed = self._remove_locked(user_id)
        if removed:
            logger.debug("Removed memory for user %s", user_id)
        return removed

    def get_all_user_ids(self) -> list[str]:
        with self._lock:
            return list(self._memories)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            lookup = self._chunk_lookup
            lookup_bytes = lookup.nbytes if lookup is not None else 0
            stats: dict[str, Any] = {
                "num_users": len(self._memories),
                "total_tokens": self._total_tokens,
                "total_gpu_mb": bytes_to_mb(self._total_bytes + lookup_bytes),
            }
            if lookup is not None:
                stats.update(
                    gpu_chunk_map_mb=bytes_to_mb(lookup_bytes),
                    chunk_map_bytes=lookup_bytes,
                    chunk_map_device=str(lookup.device),
                )
            return stats

    # Corpus access: every path below resolves IDs through the GPU lookup.

    def finalize_packed(self) -> None:
        """Entries stay as individual tensors; nothing to pack."""
        return

    def finalize_chunk_lookup(self) -> None:
        with self._lock:
            self._lookup_locked()

    def close_chunk_lookup(self) -> None:
        with self._lock:
            self._close_lookup_locked()

    def get_chunk_memories(self, chunk_ids: Sequence[str]) -> list[tuple[str, UserMemory]]:
        with self._lock:
            return self._lookup_locked().resolve(chunk_ids)

    def build_device_row_map(self, stable_id_items: Iterable[tuple[int, str]]) -> Any:
        with self._lock:
            return self._lookup_locked().bind_stable_ids(stable_id_items)

    def get_device_chunk_memories(
        self, stable_ids: Any, id_to_row: Any, *, reverse: bool = False
    ) -> list[tuple[str, UserMemory]]:
        with self._lock:
            return self._lookup_locked().resolve_device(stable_ids, id_to_row, reverse=reverse)

    # Staging and transfer metrics: no-ops for GPU-resident entries.

    def prefetch_user_to_gpu(self, user_id: str) -> bool:
        del user_id
        return False

    def release_staging(self, user_id: str) -> None:
        del user_id

    def get_bench_summary(self) -> dict[str, float | int]:
        return {}

    def transfer_totals(self) -> dict[str, float]:
        return {}

    def _lookup_locked(self) -> ChunkLookup[UserMemory]:
        if self._chunk_lookup is None:
            self._chunk_lookup = ChunkLookup(self._memories.items(), device=self._device)
        return self._chunk_lookup

    def _close_lookup_locked(self) -> None:
        if self._chunk_lookup is not None:
            self._chunk_lookup.close()
            self._chunk_lookup = None

    def _remove_locked(self, user_id: str) -> bool:
        memory = self._memories.pop(user_id, None)
        if memory is None:
            return False
        self._close_lookup_locked()
        self._total_tokens -= memory.num_tokens
        self._total_bytes -= self._bytes_by_user.pop(user_id, 0)
        del memory.kv_by_layer
        return True


def _contiguous_on_device(tensor: Any, device: str) -> Any:
    if str(tensor.device) != device:
        tensor = tensor.to(device=device)
    return tensor.contiguous()


def kv_nbytes(kv_by_layer: dict[str, Any]) -> int:
    return sum(tensor_nbytes(tensor) for tensor in kv_by_layer.values())


def tensor_nbytes(tensor: Any) -> int:
    return int(tensor.nbytes)
