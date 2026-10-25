from __future__ import annotations

from benchmarks.common.lmcache.load_stats import (
    BlendLoadStats,
    load_stats,
    reset_load_stats,
    snapshot_load_stats,
)


def test_snapshot_keeps_the_largest_match_per_request_and_sums_load_time() -> None:
    stats = BlendLoadStats()
    stats.record_match("3", 40)
    stats.record_match("3", 24)  # a preempted request is looked up again
    stats.record_match("4", 0)
    stats.record_load_time(1.5)
    stats.record_load_time(2.0)

    assert stats.snapshot() == {
        "requests_seen": 2,
        "requests_matched": 1,
        "tokens_matched": 40,
        "load_calls": 2,
        "load_time_ms": 3.5,
        "matched_tokens": {"3": 40, "4": 0},
    }


def test_reset_clears_everything() -> None:
    stats = BlendLoadStats()
    stats.record_match("1", 8)
    stats.record_load_time(1.0)

    stats.reset()

    assert stats.snapshot() == {
        "requests_seen": 0,
        "requests_matched": 0,
        "tokens_matched": 0,
        "load_calls": 0,
        "load_time_ms": 0.0,
        "matched_tokens": {},
    }


def test_module_functions_share_one_instance() -> None:
    reset_load_stats()
    load_stats.record_match("1", 5)

    assert snapshot_load_stats()["tokens_matched"] == 5

    reset_load_stats()
    assert snapshot_load_stats()["requests_seen"] == 0
