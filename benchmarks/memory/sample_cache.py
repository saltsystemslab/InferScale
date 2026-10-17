"""The precomputed composers of the samples one answer client is serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .composer import SampleComposer


@dataclass(slots=True)
class PrecomputedSample:
    composer: SampleComposer
    metrics: dict[str, Any]


class PrecomputedSamples:
    """Composers keyed by sample, with at most one sample active for answering."""

    def __init__(self) -> None:
        self._entries: dict[int, PrecomputedSample] = {}
        self._active_key: int | None = None

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def active_key(self) -> int | None:
        return self._active_key

    @property
    def active_composer(self) -> SampleComposer | None:
        if self._active_key is None:
            return None
        return self._entries[self._active_key].composer

    @property
    def active_metrics(self) -> dict[str, Any]:
        if self._active_key is None:
            return {}
        return dict(self._entries[self._active_key].metrics)

    def put(self, key: int, composer: SampleComposer, metrics: dict[str, Any]) -> None:
        self.release(key)
        self._entries[key] = PrecomputedSample(composer=composer, metrics=dict(metrics))

    def activate(self, key: int, sample_id: str) -> None:
        if self._active_key is not None and self._active_key != key:
            self.release_active()
        if key not in self._entries:
            raise RuntimeError(f"The KV cache for sample_id={sample_id} was not precomputed.")
        self._active_key = key

    def release(self, key: int) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            entry.composer.close()
        if self._active_key == key:
            self._active_key = None

    def release_active(self) -> None:
        if self._active_key is not None:
            self.release(self._active_key)

    def release_all(self) -> None:
        for key in list(self._entries):
            self.release(key)
        self._active_key = None
