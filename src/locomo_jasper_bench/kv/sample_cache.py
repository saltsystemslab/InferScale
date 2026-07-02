from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GpuSampleCacheEntry:
    composer: Any
    metrics: dict[str, Any]


class GpuSampleCacheStore:
    """Owns precomputed GPU-resident sample KV caches for one answer client."""

    def __init__(self) -> None:
        self._entries: dict[str, GpuSampleCacheEntry] = {}
        self.active_sample_key: str | None = None
        self.active_composer: Any | None = None
        self.active_metrics: dict[str, Any] = {}

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, sample_key: str, composer: Any, metrics: dict[str, Any]) -> None:
        self.release(sample_key)
        self._entries[sample_key] = GpuSampleCacheEntry(composer=composer, metrics=dict(metrics))

    def prepare(self, sample_key: str, sample_id: str) -> GpuSampleCacheEntry:
        if self.active_sample_key is not None and self.active_sample_key != sample_key:
            self.release_active()

        entry = self._entries.get(sample_key)
        if entry is None:
            raise RuntimeError(f"GPU-resident KV cache for sample_id={sample_id} was not precomputed.")

        self.active_sample_key = sample_key
        self.active_composer = entry.composer
        self.active_metrics = dict(entry.metrics)
        return entry

    def release(self, sample_key: str) -> None:
        entry = self._entries.pop(sample_key, None)
        if entry is not None:
            _close(entry.composer)

        if self.active_sample_key == sample_key:
            if entry is None:
                _close(self.active_composer)
            self.active_sample_key = None
            self.active_composer = None
            self.active_metrics = {}

    def release_active(self) -> None:
        if self.active_sample_key is not None:
            self.release(self.active_sample_key)
            return
        _close(self.active_composer)
        self.active_composer = None
        self.active_metrics = {}

    def release_all(self) -> None:
        if self.active_sample_key is not None:
            self.release_active()
        for sample_key in list(self._entries):
            self.release(sample_key)
        self.active_sample_key = None
        self.active_composer = None
        self.active_metrics = {}


def _close(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()
