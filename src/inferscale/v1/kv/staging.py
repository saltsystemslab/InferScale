"""GPU staging buffers for one memory streamed from pinned host memory."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


class LayerKVView:
    """Dict-like view over per-layer staging tensors that waits per layer.

    First access to a layer waits on that layer's H2D copy event (timing any
    blocking wait into ``stall_ms``), makes the compute stream wait on the
    event, and registers the tensor on the compute stream so the allocator
    cannot reuse it until in-flight consumers finish. ``values()`` and
    ``items()`` are lazy so iterating never syncs every layer at once.
    """

    def __init__(
        self,
        gpu_tensors: dict[str, Any],
        layer_events: dict[str, Any],
        compute_stream_fn: Callable[[], Any],
    ) -> None:
        self._gpu = gpu_tensors
        self._events = layer_events
        self._compute_stream_fn = compute_stream_fn
        self._waited: set[str] = set()
        self.stall_ms = 0.0

    def __getitem__(self, key: str) -> Any:
        tensor = self._gpu[key]
        if key not in self._waited:
            self._wait_for_layer(key, tensor)
        return tensor

    def _wait_for_layer(self, key: str, tensor: Any) -> None:
        event = self._events[key]
        compute_stream = self._compute_stream_fn()
        if not event.query():
            stall_started = time.perf_counter()
            event.synchronize()
            self.stall_ms += (time.perf_counter() - stall_started) * 1e3
        compute_stream.wait_event(event)
        tensor.record_stream(compute_stream)
        self._waited.add(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._gpu:
            return default
        return self[key]

    def __contains__(self, key: str) -> bool:
        return key in self._gpu

    def __iter__(self) -> Iterator[str]:
        return iter(self._gpu)

    def __len__(self) -> int:
        return len(self._gpu)

    def keys(self):
        return self._gpu.keys()

    def values(self) -> Iterator[Any]:
        return (self[key] for key in self._gpu)

    def items(self) -> Iterator[tuple[str, Any]]:
        return ((key, self[key]) for key in self._gpu)


@dataclass(slots=True)
class StagingSlot:
    """One memory's staged GPU buffers plus everything released with them."""

    gpu_tensors: dict[str, Any]
    start_event: Any
    end_event: Any
    view: LayerKVView
