"""Per-request accounting of what LMCache matched and how long its loads took.

LMCache's own statistics are drained by a background thread every few
seconds, so the connector records the figures the benchmark needs itself: the
matched token count the scheduler-side lookup returned for each request, and
the time the worker spent loading and blending. The engine runs in-process,
so an answer client reads a snapshot right after its request.
"""

from __future__ import annotations

import threading
from typing import Any


class BlendLoadStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._matched_tokens: dict[str, int] = {}
        self._load_calls = 0
        self._load_time_ms = 0.0

    def record_match(self, request_id: str, matched_tokens: int) -> None:
        """Keep the largest count per request; a preempted request is looked up again."""
        with self._lock:
            key = str(request_id)
            self._matched_tokens[key] = max(self._matched_tokens.get(key, 0), int(matched_tokens))

    def record_load_time(self, elapsed_ms: float) -> None:
        with self._lock:
            self._load_calls += 1
            self._load_time_ms += float(elapsed_ms)

    def reset(self) -> None:
        with self._lock:
            self._matched_tokens.clear()
            self._load_calls = 0
            self._load_time_ms = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            matched = dict(self._matched_tokens)
            return {
                "requests_seen": len(matched),
                "requests_matched": sum(1 for count in matched.values() if count > 0),
                "tokens_matched": sum(matched.values()),
                "load_calls": self._load_calls,
                "load_time_ms": self._load_time_ms,
                "matched_tokens": matched,
            }


load_stats = BlendLoadStats()


def reset_load_stats() -> None:
    load_stats.reset()


def snapshot_load_stats() -> dict[str, Any]:
    return load_stats.snapshot()
