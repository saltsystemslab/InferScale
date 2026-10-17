from __future__ import annotations

from inferscale.v1.kv.load_stats import LoadStats


def test_snapshot_counts_loads_and_covered_requests() -> None:
    stats = LoadStats()
    stats.record_load(8, "req-a")
    stats.record_load(8, "req-a")  # a preempted request reloads: same request
    stats.record_native_covered("req-b")
    stats.record_load(4)  # loads without a request id count tokens only

    assert stats.snapshot() == {
        "requests_loaded": 3,
        "tokens_loaded": 20,
        "requests_covered": 2,
    }


def test_reset_clears_every_counter() -> None:
    stats = LoadStats()
    stats.record_load(8, "req-a")
    stats.record_native_covered("req-b")

    stats.reset()

    assert stats.snapshot() == {"requests_loaded": 0, "tokens_loaded": 0, "requests_covered": 0}
