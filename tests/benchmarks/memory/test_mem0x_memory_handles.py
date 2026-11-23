from __future__ import annotations

from benchmarks.memory.mem0 import memory_handles
from benchmarks.memory.mem0.memory_handles import TimedEmbedder


class RecordingEmbedder:
    provider_name = "recording"

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def embed(self, text, memory_action=None):
        self.calls.append(("embed", text, memory_action))
        if self.failure is not None:
            raise self.failure
        return [float(len(text)), 1.0]

    def embed_batch(self, texts, memory_action="add"):
        self.calls.append(("embed_batch", list(texts), memory_action))
        if self.failure is not None:
            raise self.failure
        return [[float(len(text)), 1.0] for text in texts]


def ticking_clock(monkeypatch, seconds_per_reading):
    readings = iter(seconds_per_reading)
    monkeypatch.setattr(memory_handles.time, "perf_counter", lambda: next(readings))


def test_timed_embedder_forwards_calls_and_adds_up_their_duration(monkeypatch):
    ticking_clock(monkeypatch, [10.0, 10.25, 20.0, 20.5, 30.0, 31.0])
    wrapped = RecordingEmbedder()
    timed = TimedEmbedder(wrapped)

    assert timed.embed_time_ms == 0.0
    assert timed.embed("tea") == [3.0, 1.0]
    assert timed.embed_time_ms == 250.0
    assert timed.embed(text="coffee", memory_action="search") == [6.0, 1.0]
    assert timed.embed_time_ms == 750.0
    assert timed.embed_batch(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert timed.embed_time_ms == 1750.0
    assert wrapped.calls == [
        ("embed", "tea", None),
        ("embed", "coffee", "search"),
        ("embed_batch", ["a", "bb"], "add"),
    ]
