"""Immutable text-ID to KV-row lookup with the mapping resident on CUDA."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any


class GPUChunkMap:
    """Resolve chunk IDs on CUDA independently of where KV payloads live.

    Host code hashes the requested strings, then CUDA searches the resident
    keys and retrieves corpus rows.
    Setup may inspect IDs on the host, but the map retains no host lookup
    table and has no CPU lookup fallback.
    """

    def __init__(self, chunk_ids: Sequence[str], *, device: str) -> None:
        # Validate before importing torch so invalid targets fail clearly even
        # in lightweight environments without the optional CUDA dependencies.
        if not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
            raise ValueError("GPUChunkMap requires a CUDA device, such as cuda:0.")
        keys = _registered_keys(chunk_ids)

        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._closed = False
        self._size = len(keys)
        self._primary = torch.tensor(
            [primary for primary, _, _ in keys], dtype=torch.long, device=self._device
        )
        self._secondary = torch.tensor(
            [secondary for _, secondary, _ in keys], dtype=torch.long, device=self._device
        )
        self._rows = torch.tensor(
            [row for _, _, row in keys], dtype=torch.long, device=self._device
        )
        # Resolve an unspecified CUDA device to the device actually allocated.
        self._device = self._primary.device

    @property
    def device(self) -> Any:
        return self._device

    @property
    def nbytes(self) -> int:
        """Resident key and row tensor bytes, excluding transient query inputs."""
        if self._closed:
            return 0
        return sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (self._primary, self._secondary, self._rows)
        )

    def lookup(self, chunk_ids: Sequence[str]) -> Any:
        """Return CUDA int64 registration rows in the requested order."""
        self._require_open()
        ids = _string_ids(chunk_ids)
        torch = self._torch
        if not ids:
            return torch.empty(0, dtype=torch.long, device=self._device)
        if not self._size:
            raise ValueError("Retrieved chunk ID has no pre-encoded KV chunk.")

        keys = [_fingerprint(chunk_id) for chunk_id in ids]
        primary = torch.tensor(
            [key[0] for key in keys], dtype=torch.long, device=self._device
        )
        secondary = torch.tensor(
            [key[1] for key in keys], dtype=torch.long, device=self._device
        )
        positions = torch.searchsorted(self._primary, primary)
        safe_positions = positions.clamp(max=self._size - 1)
        valid = (
            (positions < self._size)
            & (self._primary.index_select(0, safe_positions) == primary)
            & (self._secondary.index_select(0, safe_positions) == secondary)
        )
        if not bool(valid.all().item()):
            raise ValueError("Retrieved chunk ID has no pre-encoded KV chunk.")
        return self._rows.index_select(0, safe_positions)

    def stable_id_row_map(self, bindings: Iterable[tuple[int, str]]) -> Any:
        """Bind contiguous Jasper stable IDs directly to CUDA corpus rows."""
        self._require_open()
        return self.lookup(_stable_binding_ids(bindings))

    def close(self) -> None:
        self._primary = None
        self._secondary = None
        self._rows = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The GPU chunk map is closed.")


def _fingerprint(chunk_id: str) -> tuple[int, int]:
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return (
        int.from_bytes(digest[:8], byteorder="big", signed=True),
        int.from_bytes(digest[8:16], byteorder="big", signed=True),
    )


def _string_ids(chunk_ids: Sequence[str]) -> list[str]:
    if isinstance(chunk_ids, (str, bytes)):
        raise TypeError("Chunk IDs must be a sequence of strings.")
    ids = list(chunk_ids)
    if any(not isinstance(chunk_id, str) for chunk_id in ids):
        raise TypeError("Every chunk ID must be a string.")
    return ids


def _registered_keys(chunk_ids: Sequence[str]) -> list[tuple[int, int, int]]:
    ids = _string_ids(chunk_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("GPU chunk map registration contains duplicate chunk IDs.")
    keys = sorted((*_fingerprint(chunk_id), row) for row, chunk_id in enumerate(ids))
    # Equal primary hashes would make binary search ambiguous even if the
    # secondary hashes differ.
    # Fail closed rather than selecting an arbitrary row or consulting a host
    # dictionary in this exceptionally unlikely case.
    if any(first[0] == second[0] for first, second in zip(keys, keys[1:])):
        raise ValueError("GPU chunk map registration contains a chunk-ID hash collision.")
    return keys


def _stable_binding_ids(bindings: Iterable[tuple[int, str]]) -> list[str]:
    items = list(bindings)
    if not items:
        raise ValueError("Cannot build a device row map without stable-ID bindings.")
    stable_ids: list[int] = []
    chunk_ids: list[str] = []
    for stable_id, chunk_id in items:
        if isinstance(stable_id, bool) or not isinstance(stable_id, Integral):
            raise ValueError("Jasper stable IDs must be integers.")
        if stable_id < 0:
            raise ValueError("Jasper stable IDs must be non-negative.")
        stable_ids.append(int(stable_id))
        chunk_ids.append(chunk_id)
    _string_ids(chunk_ids)
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("Jasper stable-ID bindings contain duplicates.")
    if sorted(stable_ids) != list(range(len(stable_ids))):
        raise ValueError("Jasper stable IDs must be contiguous starting at zero.")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Jasper chunk IDs must each have exactly one stable-ID binding.")
    return [chunk_id for _, chunk_id in sorted(zip(stable_ids, chunk_ids))]
