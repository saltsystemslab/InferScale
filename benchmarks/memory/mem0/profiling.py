"""Opt-in elapsed-time diagnostics around the original Mem0 retrieval code.

No retrieval algorithm is replaced. Scoped hooks retain the original callables;
adapter bindings carry the request collector into Mem0's entity worker threads.
"""

from __future__ import annotations

import functools
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Callable, Iterator


_MAIN_STAGES = (
    "lemmatization", "entity_extraction", "query_embedding", "entity_embedding",
    "primary_search", "keyword_search", "entity_boost", "candidate_build",
    "ranking", "result_format",
)
_SEARCH_STAGES = (
    "search", "adapter_prepare", "backend_search", "adapter_count", "adapter_validation",
    "payload_access", "qdrant_scoring", "qdrant_filter", "qdrant_point_construction",
    "jasper_graph_search", "jasper_exact_gpu_search", "jasper_exact_cpu_search",
)
_STAGES = tuple(dict.fromkeys((
    *_MAIN_STAGES,
    *(f"{role}_{stage}" for role in ("primary", "entity") for stage in _SEARCH_STAGES),
)))
RETRIEVAL_STAGE_METRIC_KEYS = tuple(dict.fromkeys((
    *(f"mem0_{stage}_time_ms" for stage in _STAGES),
    *(f"mem0_entity_{stage}_wall_time_ms" for stage in _SEARCH_STAGES),
    "mem0_primary_search_calls", "mem0_entity_search_calls",
    "mem0_entity_coordination_residual_time_ms", "mem0_request_residual_time_ms",
    "mem0_primary_backend_residual_wall_time_ms", "mem0_entity_backend_residual_wall_time_ms",
    "mem0_result_conversion_time_ms",
)))

Interval = tuple[int, int]


def _merged(intervals: list[Interval]) -> list[Interval]:
    result: list[Interval] = []
    for start, stop in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return result


def _duration(intervals: list[Interval]) -> float:
    return sum(stop - start for start, stop in intervals) / 1_000_000


def residual_ms(parents: list[Interval], children: list[Interval]) -> float:
    """Parent union minus child coverage, clipped to parents, never thread sums."""
    parents = _merged(parents)
    clipped = [(max(a, c), min(b, d)) for a, b in parents for c, d in _merged(children)
               if max(a, c) < min(b, d)]
    return max(0.0, _duration(parents) - _duration(_merged(clipped)))


@dataclass
class RetrievalProfile:
    memory: Any = None
    backend: str | None = None
    _spans: dict[str, list[Interval]] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.Lock)

    def add(self, spans: dict[str, list[Interval]]) -> None:
        with self._lock:
            for stage, intervals in spans.items():
                self._spans.setdefault(stage, []).extend(intervals)

    def intervals(self, stage: str) -> tuple[Interval, ...]:
        with self._lock:
            return tuple(self._spans.get(stage, ()))

    def metrics(self) -> dict[str, float]:
        # Called only after Mem0 has joined all entity workers.
        with self._lock:
            spans = {key: list(value) for key, value in self._spans.items()}
        result: dict[str, float] = {}
        for stage in _STAGES:
            if "_qdrant_" in stage and self.backend != "qdrant":
                continue
            if "_jasper_" in stage and self.backend != "jasper":
                continue
            intervals = spans.get(stage, [])
            result[f"mem0_{stage}_time_ms"] = _duration(intervals)
            if stage.startswith("entity_") and stage.removeprefix("entity_") in _SEARCH_STAGES:
                result[f"mem0_{stage}_wall_time_ms"] = _duration(_merged(intervals))
        for role in ("primary", "entity"):
            result[f"mem0_{role}_search_calls"] = float(len(spans.get(f"{role}_search", [])))
            detail = [interval for stage in _SEARCH_STAGES[5:]
                      for interval in spans.get(f"{role}_{stage}", [])]
            result[f"mem0_{role}_backend_residual_wall_time_ms"] = residual_ms(
                spans.get(f"{role}_backend_search", []), detail,
            )
        result["mem0_entity_coordination_residual_time_ms"] = residual_ms(
            spans.get("entity_boost", []),
            spans.get("entity_embedding", []) + spans.get("entity_search", []),
        )
        # Top-level stages partition the synchronous pipeline; entity_embedding
        # is nested inside entity_boost and thus unioned, not subtracted twice.
        result["mem0_request_residual_time_ms"] = residual_ms(
            spans.get("retrieval", []),
            [interval for stage in _MAIN_STAGES for interval in spans.get(stage, [])],
        )
        result["mem0_result_conversion_time_ms"] = _duration(spans.get("result_conversion", []))
        return result


@dataclass
class _Context:
    profile: RetrievalProfile
    role: str | None = None
    spans: dict[str, list[Interval]] = field(default_factory=dict)
    stack: list[str] = field(default_factory=list)
    candidate_started: int | None = None
    rank_finished: int | None = None

    def record(self, stage: str, start: int, stop: int) -> None:
        key = f"{self.role}_{stage}" if self.role else stage
        self.spans.setdefault(key, []).append((start, stop))


_active: ContextVar[_Context | None] = ContextVar("mem0_retrieval_profile", default=None)
_lock = threading.RLock()
_patches: list[tuple[Any, str, Any, Any]] = []
_users = 0
_owners: set[int] = set()


def retrieval_profiling_enabled() -> bool:
    value = os.environ.get("MEM0_PROFILE_RETRIEVAL", "0")
    if value not in {"0", "1"}:
        raise ValueError("MEM0_PROFILE_RETRIEVAL must be 0 or 1.")
    return value == "1"


@contextmanager
def profile_stage(stage: str) -> Iterator[None]:
    ctx = _active.get()
    if ctx is None:
        yield
        return
    started = perf_counter_ns()
    ctx.stack.append(stage)
    try:
        yield
    finally:
        finished = perf_counter_ns()
        ctx.stack.pop()
        ctx.record(stage, started, finished)


@contextmanager
def profile_search(adapter: Any) -> Iterator[None]:
    binding = getattr(adapter, "_retrieval_profile", None)
    if binding is None:
        # Do not accidentally profile unrelated adapters on the caller thread.
        token = _active.set(None)
        try:
            yield
        finally:
            _active.reset(token)
        return
    profile, role = binding
    ctx = _Context(profile, role)
    token = _active.set(ctx)
    try:
        with profile_stage("search"):
            yield
    finally:
        _active.reset(token)
        profile.add(ctx.spans)


def _hook(
    original: Callable[..., Any], stage: str, *, backend: bool = False, memory_method: bool = False,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        ctx = _active.get()
        enabled = ctx is not None and (
            ctx.role is not None and "backend_search" in ctx.stack if backend else ctx.role is None
        )
        if enabled and memory_method and (not args or args[0] is not ctx.profile.memory):
            token = _active.set(None)
            try:
                return original(*args, **kwargs)
            finally:
                _active.reset(token)
        if not enabled:
            return original(*args, **kwargs)
        if stage == "ranking" and ctx.candidate_started is not None:
            ctx.record("candidate_build", ctx.candidate_started, perf_counter_ns())
        with profile_stage(stage):
            result = original(*args, **kwargs)
        if stage == "ranking":
            ctx.rank_finished = perf_counter_ns()
        elif stage in {"keyword_search", "entity_boost"}:
            ctx.candidate_started = perf_counter_ns()
        return result
    return wrapped


def _pipeline_hook(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(memory: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = _active.get()
        if ctx is None:
            return original(memory, *args, **kwargs)
        if ctx.role is not None or memory is not ctx.profile.memory:
            token = _active.set(None)
            try:
                return original(memory, *args, **kwargs)
            finally:
                _active.reset(token)
        previous = ctx.candidate_started, ctx.rank_finished
        ctx.candidate_started = ctx.rank_finished = None
        try:
            result = original(memory, *args, **kwargs)
            if ctx.rank_finished is not None:
                ctx.record("result_format", ctx.rank_finished, perf_counter_ns())
            return result
        finally:
            ctx.candidate_started, ctx.rank_finished = previous
    return wrapped


def _install_hooks() -> None:
    # Imports happen only for opt-in profiling, never while importing the schema.
    import mem0.memory.main as mem0_main
    from mem0.vector_stores.base import VectorStoreBase
    from qdrant_client.local import local_collection
    from inferscale.v1.index.jasper import JasperIndex

    specs = [
        (mem0_main, "lemmatize_for_bm25", "lemmatization", False),
        (mem0_main, "extract_entities", "entity_extraction", False),
        (mem0_main, "score_and_rank", "ranking", False),
        (mem0_main.Memory, "_compute_entity_boosts", "entity_boost", False),
        (VectorStoreBase, "keyword_search", "keyword_search", False),
        (local_collection, "calculate_distance", "qdrant_scoring", True),
        (local_collection.LocalCollection, "_payload_and_non_deleted_mask", "qdrant_filter", True),
        (local_collection.LocalCollection, "_get_payload", "payload_access", True),
        (JasperIndex, "_payload_by_ordinal", "payload_access", True),
        (JasperIndex, "_search_jasper_device", "jasper_graph_search", True),
        (JasperIndex, "_search_exact_gpu_device", "jasper_exact_gpu_search", True),
        (JasperIndex, "_search_exact", "jasper_exact_cpu_search", True),
    ]
    pending: list[tuple[Any, str, Any, Any]] = []
    for owner, name, stage, backend in specs:
        original = getattr(owner, name, None)
        if not callable(original):
            raise RuntimeError(f"Mem0 retrieval profiling requires callable {name}; disable MEM0_PROFILE_RETRIEVAL.")
        pending.append((owner, name, original, _hook(
            original, stage, backend=backend, memory_method=owner is mem0_main.Memory,
        )))
    original = getattr(mem0_main.Memory, "_search_vector_store", None)
    if not callable(original):
        raise RuntimeError("Mem0 retrieval profiling requires callable _search_vector_store; disable MEM0_PROFILE_RETRIEVAL.")
    pending.append((mem0_main.Memory, "_search_vector_store", original, _pipeline_hook(original)))
    construct = getattr(local_collection, "construct", None)
    if not callable(construct):
        raise RuntimeError("Mem0 retrieval profiling requires callable construct; disable MEM0_PROFILE_RETRIEVAL.")
    timed_construct = _hook(construct, "qdrant_point_construction", backend=True)

    @functools.wraps(construct)
    def construct_point(model: Any, *args: Any, **kwargs: Any) -> Any:
        if model is local_collection.models.ScoredPoint:
            return timed_construct(model, *args, **kwargs)
        return construct(model, *args, **kwargs)

    pending.append((local_collection, "construct", construct, construct_point))
    # Validate everything before the first mutation; unwind partial installation.
    try:
        for owner, name, original, wrapper in pending:
            setattr(owner, name, wrapper)
            _patches.append((owner, name, original, wrapper))
    except BaseException:
        _restore_hooks()
        raise


def _restore_hooks() -> None:
    for owner, name, original, wrapper in reversed(_patches):
        if getattr(owner, name, None) is wrapper:
            setattr(owner, name, original)
    _patches.clear()


@contextmanager
def measure_retrieval(memory: Any, *, backend: str) -> Iterator[RetrievalProfile | None]:
    if not retrieval_profiling_enabled():
        yield None
        return
    global _users
    adapters = [(getattr(memory, "vector_store", None), "primary"),
                (getattr(memory, "_entity_store", None), "entity")]
    if any(adapter is None or not hasattr(adapter, "_retrieval_profile") for adapter, _ in adapters):
        raise RuntimeError("Mem0 retrieval profiling requires initialized benchmark primary and entity adapters.")
    if adapters[0][0] is adapters[1][0]:
        raise RuntimeError("Mem0 retrieval profiling requires distinct primary and entity adapters.")
    profile = RetrievalProfile(memory=memory, backend=backend)
    ctx = _Context(profile)
    with _lock:
        if id(memory) in _owners or any(adapter._retrieval_profile is not None for adapter, _ in adapters):
            raise RuntimeError("Concurrent retrieval profiling requests cannot share a Mem0 instance or adapter.")
        if _users == 0:
            _install_hooks()
        elif any(getattr(owner, name, None) is not wrapper for owner, name, _, wrapper in _patches):
            raise RuntimeError("Mem0 retrieval profiling hooks changed during profiling.")
        _owners.add(id(memory))
        _users += 1
        for adapter, role in adapters:
            adapter._retrieval_profile = (profile, role)
    token = _active.set(ctx)
    try:
        yield profile
    finally:
        _active.reset(token)
        profile.add(ctx.spans)
        with _lock:
            for adapter, _ in adapters:
                adapter._retrieval_profile = None
            _owners.remove(id(memory))
            _users -= 1
            if _users == 0:
                _restore_hooks()
