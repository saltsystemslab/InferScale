from __future__ import annotations

import logging
from typing import Any, Iterable

from .gpu_memory_store import GPUMemoryStore, UserMemory, bytes_to_mb, kv_nbytes

logger = logging.getLogger(__name__)


class DeviceChunkSelectionError(RuntimeError):
    """A device result cannot be mapped safely to packed chunk rows."""


class PackedGPUMemoryStore(GPUMemoryStore):
    """GPU chunk store with a packed tensor layout for device-side selection.

    Before finalization this has the same string-keyed behavior as
    ``GPUMemoryStore``. Finalization packs each layer along its token axis and
    replaces the per-fact tensors with views into the packed slabs, preserving
    the ordinary lookup path without retaining a second copy of the KV corpus.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        super().__init__(device=device)
        self._packed = False
        self._packed_kv_by_layer: dict[str, Any] = {}
        self._packed_token_ids: Any = None
        self._packed_offsets: Any = None
        self._packed_lengths: Any = None
        self._row_by_user_id: dict[str, int] = {}
        self._max_tokens_per_memory = 0
        self._packed_physical_bytes = 0

    @property
    def is_packed(self) -> bool:
        return self._packed

    def add_user_memory(
        self,
        user_id: str,
        kv_by_layer: dict[str, Any],
        num_tokens: int,
        token_ids: list[int] | None = None,
    ) -> None:
        if self._packed:
            raise RuntimeError(
                "Cannot add chunks after the packed GPU store is finalized."
            )
        super().add_user_memory(
            user_id=user_id,
            kv_by_layer=kv_by_layer,
            num_tokens=num_tokens,
            token_ids=token_ids,
        )

    def finalize_packed(self) -> None:
        """Freeze registration and build one packed token slab per KV layer."""
        import torch

        with self._lock:
            if self._packed:
                return
            user_ids = list(self._memories)
            if not user_ids:
                self._packed = True
                return

            memories = [self._memories[user_id] for user_id in user_ids]
            layer_names = _validate_packable_memories(memories)
            lengths = [memory.num_tokens for memory in memories]
            offsets: list[int] = []
            next_offset = 0
            for length in lengths:
                offsets.append(next_offset)
                next_offset += length

            packed_kv_by_layer: dict[str, Any] = {}
            try:
                for layer_name in layer_names:
                    sources = [memory.kv_by_layer[layer_name] for memory in memories]
                    packed_kv_by_layer[layer_name] = torch.cat(sources, dim=1)
                    for memory in memories:
                        memory.kv_by_layer.pop(layer_name)
                    del sources
            except Exception:
                # Completed slabs can restore the per-fact lookup views, so
                # a later-layer allocation failure does not corrupt fallback.
                for memory, offset, length in zip(memories, offsets, lengths):
                    for layer_name, packed in packed_kv_by_layer.items():
                        memory.kv_by_layer[layer_name] = packed[
                            :, offset : offset + length
                        ]
                    memory.kv_by_layer = {
                        layer_name: memory.kv_by_layer[layer_name]
                        for layer_name in layer_names
                    }
                raise

            try:
                flat_token_ids = [
                    token_id
                    for memory in memories
                    for token_id in (memory.token_ids or ())
                ]
                packed_token_ids = torch.tensor(
                    flat_token_ids,
                    device=self._device,
                    dtype=torch.long,
                )
                packed_offsets = torch.tensor(
                    offsets,
                    device=self._device,
                    dtype=torch.long,
                )
                packed_lengths = torch.tensor(
                    lengths,
                    device=self._device,
                    dtype=torch.long,
                )
            except Exception:
                for memory, offset, length in zip(memories, offsets, lengths):
                    memory.kv_by_layer = {
                        layer_name: packed[:, offset : offset + length]
                        for layer_name, packed in packed_kv_by_layer.items()
                    }
                raise

            for row, (user_id, memory, offset, length) in enumerate(
                zip(user_ids, memories, offsets, lengths)
            ):
                memory.kv_by_layer = {
                    layer_name: packed[:, offset : offset + length]
                    for layer_name, packed in packed_kv_by_layer.items()
                }
                self._row_by_user_id[user_id] = row

            self._packed_kv_by_layer = packed_kv_by_layer
            self._packed_token_ids = packed_token_ids
            self._packed_offsets = packed_offsets
            self._packed_lengths = packed_lengths
            self._max_tokens_per_memory = max(lengths)
            self._packed_physical_bytes = (
                kv_nbytes(packed_kv_by_layer)
                + _tensor_nbytes(packed_token_ids)
                + _tensor_nbytes(packed_offsets)
                + _tensor_nbytes(packed_lengths)
            )
            self._packed = True
        logger.info(
            "Packed %d GPU memory chunks across %d layers.",
            len(user_ids),
            len(layer_names),
        )

    def build_device_row_map(
        self,
        stable_id_items: Iterable[tuple[int, str]],
    ) -> Any:
        """Build a CUDA stable-ID to packed-row lookup for one Jasper store.

        The returned map is valid while the finalized corpus is immutable.
        The throughput path removes rows only when every request is complete.
        """
        import torch

        bindings = [
            (int(stable_id), str(user_id)) for stable_id, user_id in stable_id_items
        ]
        if not bindings:
            raise ValueError(
                "Cannot build a device row map without stable-ID bindings."
            )
        stable_ids = [stable_id for stable_id, _ in bindings]
        if any(stable_id < 0 for stable_id in stable_ids):
            raise ValueError("Jasper stable IDs must be non-negative.")
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("Jasper stable-ID bindings contain duplicates.")
        if sorted(stable_ids) != list(range(len(stable_ids))):
            raise ValueError(
                "Packed GPU selection requires contiguous Jasper stable IDs starting at zero."
            )

        with self._lock:
            if not self._packed:
                raise RuntimeError(
                    "The GPU chunk store must be finalized before row-map binding."
                )
            row_by_user_id = dict(self._row_by_user_id)
        dense_rows = [-1] * len(stable_ids)
        seen_user_ids: set[str] = set()
        for stable_id, user_id in bindings:
            if user_id in seen_user_ids:
                raise ValueError(f"Jasper item ID {user_id!r} is bound more than once.")
            seen_user_ids.add(user_id)
            row = row_by_user_id.get(user_id)
            if row is None:
                raise ValueError(
                    f"Jasper item ID {user_id!r} has no packed GPU KV chunk."
                )
            dense_rows[stable_id] = row
        return torch.tensor(dense_rows, device=self._device, dtype=torch.long)

    def select_device_ids(
        self,
        stable_ids: Any,
        id_to_row: Any,
        *,
        reverse: bool,
    ) -> UserMemory:
        """Gather variable-length chunks without materializing Jasper IDs on the host."""
        import torch

        if int(getattr(stable_ids, "ndim", -1)) != 1:
            raise DeviceChunkSelectionError(
                "Jasper stable IDs must be a one-dimensional tensor."
            )
        if int(stable_ids.numel()) < 1:
            raise DeviceChunkSelectionError("Jasper returned no stable IDs.")
        if int(getattr(id_to_row, "ndim", -1)) != 1 or int(id_to_row.numel()) < 1:
            raise DeviceChunkSelectionError(
                "The Jasper stable-ID row map is empty or invalid."
            )
        if getattr(stable_ids, "device", None) != getattr(id_to_row, "device", None):
            raise DeviceChunkSelectionError(
                "Jasper stable IDs and the KV row map must be on the same device."
            )

        stable_ids = stable_ids.to(dtype=torch.long)
        valid_range = (stable_ids >= 0) & (stable_ids < int(id_to_row.numel()))
        safe_ids = torch.where(valid_range, stable_ids, torch.zeros_like(stable_ids))
        rows = id_to_row.index_select(0, safe_ids)
        valid = (valid_range & (rows >= 0)).all()
        if int(stable_ids.numel()) > 1:
            sorted_ids = torch.sort(stable_ids).values
            valid = valid & (sorted_ids[1:] != sorted_ids[:-1]).all()
        if not bool(valid.item()):
            raise DeviceChunkSelectionError(
                "Jasper returned a padded, invalid, unmapped, or duplicate stable ID."
            )

        if reverse:
            rows = torch.flip(rows, dims=(0,))

        with self._lock:
            if not self._packed:
                raise RuntimeError(
                    "The GPU chunk store must be finalized before selection."
                )
            layer_names = tuple(self._packed_kv_by_layer)
            packed_token_ids = self._packed_token_ids
            packed_offsets = self._packed_offsets
            packed_lengths = self._packed_lengths
            max_tokens = self._max_tokens_per_memory

        selected_offsets = packed_offsets.index_select(0, rows)
        selected_lengths = packed_lengths.index_select(0, rows)
        positions = torch.arange(
            max_tokens,
            device=rows.device,
            dtype=packed_offsets.dtype,
        )
        position_grid = positions.unsqueeze(0)
        valid_positions = position_grid < selected_lengths.unsqueeze(1)
        source_indices = (selected_offsets.unsqueeze(1) + position_grid).masked_select(
            valid_positions
        )
        selected_token_ids = packed_token_ids.index_select(0, source_indices)
        token_ids = selected_token_ids.detach().cpu().tolist()
        return UserMemory(
            # Match the CPU store's lazy layer view: composition gathers one
            # layer at a time instead of retaining a selected copy of every
            # layer alongside the packed corpus.
            kv_by_layer=_PackedSelectionLayerView(
                self,
                layer_names,
                source_indices,
            ),
            num_tokens=len(token_ids),
            token_ids=token_ids,
        )

    def remove_user_memory(self, user_id: str) -> bool:
        removed = super().remove_user_memory(user_id)
        if not removed:
            return False
        with self._lock:
            self._row_by_user_id.pop(user_id, None)
            if not self._memories:
                self._clear_packed_locked()
        return True

    def get_stats(self) -> dict[str, int | float]:
        with self._lock:
            physical_bytes = (
                self._packed_physical_bytes
                if self._packed and self._memories
                else self._total_bytes
            )
            return {
                "num_users": len(self._memories),
                "total_tokens": self._total_tokens,
                "total_gpu_mb": bytes_to_mb(physical_bytes),
            }

    def _gather_packed_layer(self, layer_name: str, source_indices: Any) -> Any:
        with self._lock:
            packed = self._packed_kv_by_layer.get(layer_name)
            if packed is None:
                raise RuntimeError(
                    f"Packed GPU layer {layer_name!r} is unavailable after store close."
                )
        return packed.index_select(1, source_indices)

    def _clear_packed_locked(self) -> None:
        self._packed_kv_by_layer.clear()
        self._packed_token_ids = None
        self._packed_offsets = None
        self._packed_lengths = None
        self._row_by_user_id.clear()
        self._max_tokens_per_memory = 0
        self._packed_physical_bytes = 0


def _validate_packable_memories(memories: list[UserMemory]) -> tuple[str, ...]:
    if not memories:
        raise ValueError("Cannot pack an empty memory list.")
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
            raise ValueError(
                "Packed GPU chunks must have identical ordered layer names."
            )
        for layer_name in layer_names:
            schema = _layer_schema(memory.kv_by_layer[layer_name], memory.num_tokens)
            if schema != expected_schema[layer_name]:
                raise ValueError(
                    f"Packed GPU chunk schema mismatch for layer {layer_name!r}."
                )
    return layer_names


class _PackedSelectionLayerView:
    def __init__(
        self,
        store: PackedGPUMemoryStore,
        layer_names: tuple[str, ...],
        source_indices: Any,
    ) -> None:
        self._store = store
        self._layer_names = layer_names
        self._source_indices = source_indices

    def __getitem__(self, layer_name: str) -> Any:
        if layer_name not in self._layer_names:
            raise KeyError(layer_name)
        return self._store._gather_packed_layer(
            layer_name,
            self._source_indices,
        )

    def __iter__(self):
        return iter(self._layer_names)

    def __len__(self) -> int:
        return len(self._layer_names)

    def keys(self):
        return self._layer_names


def _layer_schema(tensor: Any, num_tokens: int) -> tuple[Any, ...]:
    shape = tuple(getattr(tensor, "shape", ()))
    if len(shape) < 2 or int(shape[0]) != 2 or int(shape[1]) != num_tokens:
        raise ValueError(
            "Packed GPU KV tensors must have shape [2, tokens, ...] matching num_tokens."
        )
    return (
        getattr(tensor, "dtype", None),
        getattr(tensor, "device", None),
        *shape[2:],
    )


def _tensor_nbytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    element_size = getattr(tensor, "element_size", None)
    nelement = getattr(tensor, "nelement", None)
    if callable(element_size) and callable(nelement):
        return int(element_size() * nelement())
    return 0
