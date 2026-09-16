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
    query_elapsed_ns: int = 0
    query_calls: int = 0
    query_intervals_ns: list[tuple[int, int]] | None = None

    @property
    def time_ms(self) -> float:
        return self.elapsed_ns / 1_000_000

    @property
    def query_time_ms(self) -> float:
        return self.query_elapsed_ns / 1_000_000


@dataclass(slots=True)
class DeepcopyTotals:
    """Sum copy durations and merge their wall intervals across joined workers."""

    elapsed_ns: int = 0
    calls: int = 0
    _intervals_ns: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)
    query_elapsed_ns: int = 0
    query_calls: int = 0
    _query_intervals_ns: list[tuple[int, int]] = field(default_factory=list, repr=False)

    def add(self, timing: DeepcopyTiming) -> None:
        if timing.intervals_ns is None:
            raise ValueError("Aggregated deepcopy timing requires recorded intervals.")
        if timing.query_intervals_ns is None and (timing.query_calls or timing.query_elapsed_ns):
            raise ValueError("Aggregated query deepcopy timing requires recorded intervals.")
        # Each worker owns its timing until it publishes at query completion.
        with self._lock:
            self.elapsed_ns += timing.elapsed_ns
            self.calls += timing.calls
            self._intervals_ns.extend(timing.intervals_ns)
            self.query_elapsed_ns += timing.query_elapsed_ns
            self.query_calls += timing.query_calls
            self._query_intervals_ns.extend(timing.query_intervals_ns or ())

    @property
    def time_ms(self) -> float:
        return self.elapsed_ns / 1_000_000

    @property
    def query_time_ms(self) -> float:
        return self.query_elapsed_ns / 1_000_000

    @property
    def wall_time_ms(self) -> float:
        with self._lock:
            intervals = list(self._intervals_ns)
        return _wall_time_ms(intervals)

    @property
    def query_wall_time_ms(self) -> float:
        with self._lock:
            intervals = list(self._query_intervals_ns)
        return _wall_time_ms(intervals)


def _wall_time_ms(intervals: list[tuple[int, int]]) -> float:
    elapsed_ns = 0
    end = 0
    for start, stop in sorted(intervals):
        elapsed_ns += max(0, stop - max(start, end))
        end = max(end, stop)
    return elapsed_ns / 1_000_000


@dataclass(slots=True)
class _CopyHook:
    module: Any
    original: Callable[..., Any]
    wrapper: Callable[..., Any]


@dataclass(slots=True)
class _Patch:
    hooks: tuple[_CopyHook, ...]
    users: int = 0


_active: ContextVar[DeepcopyTiming | None] = ContextVar("qdrant_deepcopy_timing", default=None)
_lock = threading.Lock()
_patch: _Patch | None = None


def deepcopy_profiling_enabled() -> bool:
    value = os.environ.get("QDRANT_PROFILE_DEEPCOPY", "0")
    if value not in {"0", "1"}:
        raise ValueError("QDRANT_PROFILE_DEEPCOPY must be 0 or 1.")
    return value == "1"


def _timed_copy(original: Callable[..., Any], *, query: bool = False) -> Callable[..., Any]:
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
            if query:
                timing.query_elapsed_ns += finished - started
                timing.query_calls += 1
                intervals = timing.query_intervals_ns
            else:
                timing.elapsed_ns += finished - started
                timing.calls += 1
                intervals = timing.intervals_ns
            if intervals is not None:
                intervals.append((started, finished))

    return wrapper


@contextmanager
def measure_qdrant_deepcopy(*, record_intervals: bool = False) -> Iterator[DeepcopyTiming]:
    """Time the original calls, preserving results, errors, and concurrent queries.

    The hooks are scoped to active diagnostics and restored on exit. A context-local
    collector separates concurrent or nested queries without serializing them.
    local_collection.deepcopy measures payload copies; qdrant_local.deepcopy
    measures query-object copies separately. Neither measures all payload work.
    """
    global _patch
    from qdrant_client.local import local_collection, qdrant_local

    with _lock:
        if _patch is None:
            hooks = []
            for module, query in ((local_collection, False), (qdrant_local, True)):
                original = getattr(module, "deepcopy", None)
                if not callable(original):
                    raise RuntimeError(
                        f"This qdrant-client version has no local deepcopy hook in {module.__name__}; "
                        "disable QDRANT_PROFILE_DEEPCOPY or use a supported client version."
                    )
                hooks.append(_CopyHook(module, original, _timed_copy(original, query=query)))
            # Validate both aliases before installing either. Roll back an
            # interrupted installation rather than leave a partial global hook.
            try:
                for hook in hooks:
                    hook.module.deepcopy = hook.wrapper
            except BaseException:
                for hook in reversed(hooks):
                    if getattr(hook.module, "deepcopy", None) is hook.wrapper:
                        hook.module.deepcopy = hook.original
                raise
            _patch = _Patch(tuple(hooks))
        elif any(getattr(hook.module, "deepcopy", None) is not hook.wrapper for hook in _patch.hooks):
            raise RuntimeError("Qdrant's deepcopy hook changed during profiling.")
        patch = _patch
        patch.users += 1

    timing = DeepcopyTiming(
        intervals_ns=[] if record_intervals else None,
        query_intervals_ns=[] if record_intervals else None,
    )
    token = _active.set(timing)
    try:
        yield timing
    finally:
        _active.reset(token)
        with _lock:
            patch.users -= 1
            if patch.users == 0:
                for hook in reversed(patch.hooks):
                    if getattr(hook.module, "deepcopy", None) is hook.wrapper:
                        hook.module.deepcopy = hook.original
                _patch = None
