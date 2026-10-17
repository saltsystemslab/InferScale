"""GPU chunk store with one packed slab per layer for device-side selection.

Finalization concatenates every chunk's KV along the token axis, one slab per
layer, and replaces the per-chunk tensors with views into those slabs. The
ordinary lookup path keeps working through the views while Jasper results can
gather their chunks on the GPU without materializing IDs on the host.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from ..units import bytes_to_mb
from .memory_store import GPUMemoryStore, UserMemory, kv_nbytes, tensor_nbytes

logger = logging.getLogger(__name__)


class PackedGPUMemoryStore(GPUMemoryStore):
    """GPU chunk store that packs the corpus at finalization."""

    def __init__(self, device: str = "cuda:0") -> None:
        super().__init__(device=device)
        self._packed = False
        self._packed_kv_by_layer: dict[str, Any] = {}
        self._packed_token_ids: Any = None
        self._packed_offsets: Any = None
        self._packed_lengths: Any = None
        self._packed_rows_valid = True
        self._max_tokens_per_memory = 0
        self._packed_physical_bytes = 0

    def add_user_memory(
        self,
        user_id: str,
        kv_by_layer: dict[str, Any],
        num_tokens: int,
        token_ids: list[int] | None = None,
    ) -> None:
        if self._packed:
            raise RuntimeError("Cannot add chunks after the packed GPU store is finalized.")
        super().add_user_memory(
            user_id=user_id, kv_by_layer=kv_by_layer, num_tokens=num_tokens, token_ids=token_ids
        )

    def finalize_packed(self) -> None:
        """Freeze registration and build one packed token slab per KV layer."""
        with self._lock:
            if self._packed:
                return
            self._lookup_locked()
            memories = list(self._memories.values())
            if not memories:
                self._packed = True
                return
            layer_names = _validate_packable_memories(memories)
            lengths = [memory.num_tokens for memory in memories]
            offsets = _prefix_offsets(lengths)
            packed = _pack_layers(memories, layer_names, offsets, lengths)
            try:
                index_tensors = _index_tensors(memories, offsets, lengths, self._device)
            finally:
                # Whether or not the index tensors could be allocated, the
                # stored payloads must keep pointing at the packed slabs.
                _restore_views(memories, packed, layer_names, offsets, lengths)
            self._packed_kv_by_layer = packed
            self._packed_token_ids, self._packed_offsets, self._packed_lengths = index_tensors
            self._max_tokens_per_memory = max(lengths)
            self._packed_physical_bytes = kv_nbytes(packed) + sum(
                tensor_nbytes(tensor) for tensor in index_tensors
            )
            self._packed = True
        logger.info(
            "Packed %d GPU memory chunks across %d layers.", len(memories), len(layer_names)
        )

    def build_device_row_map(self, stable_id_items: Iterable[tuple[int, str]]) -> Any:
        """CUDA stable-ID to packed-row map, valid while the corpus is immutable."""
        with self._lock:
            self._require_packed_locked()
            return self._lookup_locked().bind_stable_ids(stable_id_items)

    def select_device_ids(self, stable_ids: Any, id_to_row: Any, *, reverse: bool) -> UserMemory:
        """Gather variable-length chunks without materializing Jasper IDs on the host."""
        with self._lock:
            self._require_packed_locked()
            rows = self._lookup_locked().device_rows(stable_ids, id_to_row, reverse=reverse)
            layer_names = tuple(self._packed_kv_by_layer)
            packed_token_ids = self._packed_token_ids
            packed_offsets = self._packed_offsets
            packed_lengths = self._packed_lengths
            max_tokens = self._max_tokens_per_memory

        source_indices = _gather_source_indices(rows, packed_offsets, packed_lengths, max_tokens)
        token_ids = packed_token_ids.index_select(0, source_indices).detach().cpu().tolist()
        return UserMemory(
            # Like the pinned store's lazy view, composition gathers one layer
            # at a time instead of copying every selected layer up front.
            kv_by_layer=_PackedSelectionLayerView(self, layer_names, source_indices),
            num_tokens=len(token_ids),
            token_ids=token_ids,
        )

    def remove_user_memory(self, user_id: str) -> bool:
        removed = super().remove_user_memory(user_id)
        if not removed:
            return False
        with self._lock:
            if self._packed:
                self._packed_rows_valid = False
            if not self._memories:
                self._clear_packed_locked()
        return True

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            lookup = self._chunk_lookup
            lookup_bytes = lookup.nbytes if lookup is not None else 0
            physical_bytes = (
                self._packed_physical_bytes
                if self._packed and self._memories
                else self._total_bytes
            )
            return {
                "num_users": len(self._memories),
                "total_tokens": self._total_tokens,
                "total_gpu_mb": bytes_to_mb(physical_bytes + lookup_bytes),
                "gpu_chunk_map_mb": bytes_to_mb(lookup_bytes),
                "chunk_map_bytes": lookup_bytes,
                "chunk_map_device": str(lookup.device) if lookup is not None else None,
            }

    def _gather_packed_layer(self, layer_name: str, source_indices: Any) -> Any:
        with self._lock:
            packed = self._packed_kv_by_layer.get(layer_name)
            if packed is None:
                raise RuntimeError(
                    f"Packed GPU layer {layer_name!r} is unavailable after store close."
                )
        return packed.index_select(1, source_indices)

    def _require_packed_locked(self) -> None:
        if not self._packed:
            raise RuntimeError("The GPU chunk store must be finalized before selection.")
        if not self._packed_rows_valid:
            raise RuntimeError("Packed GPU row bindings are invalid after chunk removal.")

    def _clear_packed_locked(self) -> None:
        self._packed_kv_by_layer.clear()
        self._packed_token_ids = None
        self._packed_offsets = None
        self._packed_lengths = None
        self._max_tokens_per_memory = 0
        self._packed_physical_bytes = 0


class _PackedSelectionLayerView:
    """Dict-like view that gathers a selected layer from the slabs on access."""

    def __init__(
        self, store: PackedGPUMemoryStore, layer_names: tuple[str, ...], source_indices: Any
    ) -> None:
        self._store = store
        self._layer_names = layer_names
        self._source_indices = source_indices

    def __getitem__(self, layer_name: str) -> Any:
        if layer_name not in self._layer_names:
            raise KeyError(layer_name)
        return self._store._gather_packed_layer(layer_name, self._source_indices)

    def __iter__(self):
        return iter(self._layer_names)

    def __len__(self) -> int:
        return len(self._layer_names)

    def keys(self):
        return self._layer_names


def _validate_packable_memories(memories: list[UserMemory]) -> tuple[str, ...]:
    first = memories[0]
    layer_names = tuple(first.kv_by_layer)
    if not layer_names:
        raise ValueError("Cannot pack memories without KV layers.")
    expected_schema = {
        layer_name: _layer_schema(first.kv_by_layer[layer_name], first.num_tokens)
        for layer_name in layer_names
    }
    for memory in memories:
        if memory.num_tokens < 1:
            raise ValueError("Packed GPU chunks must contain at least one token.")
        if memory.token_ids is None or len(memory.token_ids) != memory.num_tokens:
            raise ValueError("Packed GPU chunks require one token ID per KV token.")
        if tuple(memory.kv_by_layer) != layer_names:
            raise ValueError("Packed GPU chunks must have identical ordered layer names.")
        for layer_name in layer_names:
            schema = _layer_schema(memory.kv_by_layer[layer_name], memory.num_tokens)
            if schema != expected_schema[layer_name]:
                raise ValueError(f"Packed GPU chunk schema mismatch for layer {layer_name!r}.")
    return layer_names


def _layer_schema(tensor: Any, num_tokens: int) -> tuple[Any, ...]:
    shape = tuple(tensor.shape)
    if len(shape) < 2 or int(shape[0]) != 2 or int(shape[1]) != num_tokens:
        raise ValueError(
            "Packed GPU KV tensors must have shape [2, tokens, ...] matching num_tokens."
        )
    return (tensor.dtype, tensor.device, *shape[2:])


def _prefix_offsets(lengths: list[int]) -> list[int]:
    offsets: list[int] = []
    next_offset = 0
    for length in lengths:
        offsets.append(next_offset)
        next_offset += length
    return offsets


def _pack_layers(
    memories: list[UserMemory],
    layer_names: tuple[str, ...],
    offsets: list[int],
    lengths: list[int],
) -> dict[str, Any]:
    """Concatenate each layer across chunks, releasing per-chunk tensors as it goes.

    If a later layer fails to allocate, the layers already packed are handed
    back to the chunks as slab views so no payload is lost.
    """
    import torch

    packed: dict[str, Any] = {}
    try:
        for layer_name in layer_names:
            packed[layer_name] = torch.cat(
                [memory.kv_by_layer[layer_name] for memory in memories], dim=1
            )
            for memory in memories:
                memory.kv_by_layer.pop(layer_name)
    except Exception:
        _restore_views(memories, packed, layer_names, offsets, lengths)
        raise
    return packed


def _restore_views(
    memories: list[UserMemory],
    packed: dict[str, Any],
    layer_names: tuple[str, ...],
    offsets: list[int],
    lengths: list[int],
) -> None:
    """Point each chunk's packed layers at slab views; unpacked layers stay as they are."""
    for memory, offset, length in zip(memories, offsets, lengths):
        memory.kv_by_layer = {
            layer_name: (
                packed[layer_name][:, offset : offset + length]
                if layer_name in packed
                else memory.kv_by_layer[layer_name]
            )
            for layer_name in layer_names
        }


def _index_tensors(
    memories: list[UserMemory], offsets: list[int], lengths: list[int], device: str
) -> tuple[Any, Any, Any]:
    """Flat token ids plus per-chunk offsets and lengths, as CUDA tensors."""
    import torch

    flat_token_ids = [token_id for memory in memories for token_id in (memory.token_ids or ())]
    return (
        torch.tensor(flat_token_ids, device=device, dtype=torch.long),
        torch.tensor(offsets, device=device, dtype=torch.long),
        torch.tensor(lengths, device=device, dtype=torch.long),
    )


def _gather_source_indices(rows: Any, offsets: Any, lengths: Any, max_tokens: int) -> Any:
    """Flat slab positions of the selected rows' tokens, in row order."""
    import torch

    selected_offsets = offsets.index_select(0, rows)
    selected_lengths = lengths.index_select(0, rows)
    positions = torch.arange(max_tokens, device=rows.device, dtype=offsets.dtype).unsqueeze(0)
    valid_positions = positions < selected_lengths.unsqueeze(1)
    return (selected_offsets.unsqueeze(1) + positions).masked_select(valid_positions)
