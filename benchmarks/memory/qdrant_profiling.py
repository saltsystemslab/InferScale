"""Opt-in timing of Qdrant local's existing deepcopy calls during a query."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Callable, Iterator


@dataclass(slots=True)
class DeepcopyTiming:
    elapsed_ns: int = 0
    calls: int = 0
    intervals_ns: list[tuple[int, int]] | None = None

    @property
    def time_ms(self) -> float:
        return self.elapsed_ns / 1_000_000


@dataclass(slots=True)
class DeepcopyTotals:
    """Sum copy durations and merge their wall intervals across joined workers."""

    elapsed_ns: int = 0
    calls: int = 0
    _intervals_ns: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def add(self, timing: DeepcopyTiming) -> None:
        if timing.intervals_ns is None:
            raise ValueError("Aggregated deepcopy timing requires recorded intervals.")
        # Each worker owns its timing until it publishes at query completion.
        with self._lock:
            self.elapsed_ns += timing.elapsed_ns
            self.calls += timing.calls
            self._intervals_ns.extend(timing.intervals_ns)

    @property
    def time_ms(self) -> float:
        return self.elapsed_ns / 1_000_000

    @property
    def wall_time_ms(self) -> float:
        with self._lock:
            intervals = sorted(self._intervals_ns)
        elapsed_ns = 0
        end = 0
        for start, stop in intervals:
            elapsed_ns += max(0, stop - max(start, end))
            end = max(end, stop)
        return elapsed_ns / 1_000_000


@dataclass(slots=True)
class _Patch:
    module: Any
    original: Callable[..., Any]
    wrapper: Callable[..., Any]
    users: int = 0


_active: ContextVar[DeepcopyTiming | None] = ContextVar("qdrant_deepcopy_timing", default=None)
_lock = threading.Lock()
_patch: _Patch | None = None


def deepcopy_profiling_enabled() -> bool:
    value = os.environ.get("QDRANT_PROFILE_DEEPCOPY", "0")
    if value not in {"0", "1"}:
        raise ValueError("QDRANT_PROFILE_DEEPCOPY must be 0 or 1.")
    return value == "1"


def _timed_copy(original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timing = _active.get()
        if timing is None:
            return original(*args, **kwargs)
        started = perf_counter_ns()
        try:
            # Only Qdrant's imported alias is wrapped. Recursive calls inside
            # copy.deepcopy remain untouched and are counted once inclusively.
            return original(*args, **kwargs)
        finally:
            finished = perf_counter_ns()
            timing.elapsed_ns += finished - started
            timing.calls += 1
            if timing.intervals_ns is not None:
                timing.intervals_ns.append((started, finished))

    return wrapper


@contextmanager
def measure_qdrant_deepcopy(*, record_intervals: bool = False) -> Iterator[DeepcopyTiming]:
    """Time the original calls, preserving results, errors, and concurrent queries.

    The hook is scoped to active diagnostics and restored on exit. A context-local
    collector separates concurrent or nested queries without serializing them.
    It measures calls through local_collection.deepcopy, not all payload work.
    """
    global _patch
    from qdrant_client.local import local_collection

    with _lock:
        if _patch is None:
            original = getattr(local_collection, "deepcopy", None)
            if not callable(original):
                raise RuntimeError(
                    "This qdrant-client version has no local deepcopy hook; "
                    "disable QDRANT_PROFILE_DEEPCOPY or use a supported client version."
                )
            _patch = _Patch(local_collection, original, _timed_copy(original))
            local_collection.deepcopy = _patch.wrapper
        elif local_collection.deepcopy is not _patch.wrapper:
            raise RuntimeError("Qdrant's deepcopy hook changed during profiling.")
        patch = _patch
        patch.users += 1

    timing = DeepcopyTiming(intervals_ns=[] if record_intervals else None)
    token = _active.set(timing)
    try:
        yield timing
    finally:
        _active.reset(token)
        with _lock:
            patch.users -= 1
            if patch.users == 0:
                if patch.module.deepcopy is patch.wrapper:
                    patch.module.deepcopy = patch.original
                _patch = None
