"""GPU-resident resolution of chunk IDs to rows of a frozen corpus snapshot.

Both chunk stores keep their payloads in a dict keyed by chunk ID, but
serving never looks a chunk up by string on the host. A ChunkLookup takes a
snapshot of the store's entries at finalization, loads their IDs into a
GPUChunkMap, and resolves string IDs or Jasper stable IDs to rows on CUDA.
Row maps bound through ``bind_stable_ids`` are tracked by weak reference so
a map built for an earlier snapshot is rejected after the corpus changes.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Sequence
from typing import Any, Generic, TypeVar

from .chunk_map import GPUChunkMap

Entry = TypeVar("Entry")


class DeviceChunkSelectionError(RuntimeError):
    """A device result cannot be mapped safely to corpus chunk rows."""


class ChunkLookup(Generic[Entry]):
    """Chunk IDs of one corpus snapshot, resolvable to entries on CUDA."""

    def __init__(self, entries: Iterable[tuple[str, Entry]], *, device: str) -> None:
        self.entries: tuple[tuple[str, Entry], ...] = tuple(entries)
        self._map = GPUChunkMap([chunk_id for chunk_id, _ in self.entries], device=device)
        self._row_maps: dict[int, weakref.ref] = {}

    @property
    def device(self) -> Any:
        return self._map.device

    @property
    def nbytes(self) -> int:
        return self._map.nbytes

    def resolve(self, chunk_ids: Sequence[str]) -> list[tuple[str, Entry]]:
        """Entries for the requested IDs, in request order."""
        rows = self._map.lookup(chunk_ids).detach().cpu().tolist()
        return [self.entries[row] for row in rows]

    def bind_stable_ids(self, stable_id_items: Iterable[tuple[int, str]]) -> Any:
        """CUDA row map from contiguous Jasper stable IDs, valid for this snapshot."""
        rows = self._map.stable_id_row_map(stable_id_items)
        self._row_maps[id(rows)] = weakref.ref(rows)
        return rows

    def device_rows(self, stable_ids: Any, id_to_row: Any, *, reverse: bool) -> Any:
        """Validated CUDA rows for a device search result."""
        binding = self._row_maps.get(id(id_to_row))
        if binding is None or binding() is not id_to_row:
            raise DeviceChunkSelectionError("The KV row map is not bound to the current corpus.")
        return device_chunk_rows(
            stable_ids,
            id_to_row,
            device=self.device,
            num_rows=len(self.entries),
            reverse=reverse,
        )

    def resolve_device(
        self, stable_ids: Any, id_to_row: Any, *, reverse: bool
    ) -> list[tuple[str, Entry]]:
        rows = self.device_rows(stable_ids, id_to_row, reverse=reverse).detach().cpu().tolist()
        return [self.entries[row] for row in rows]

    def close(self) -> None:
        self._map.close()
        self._row_maps.clear()


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
