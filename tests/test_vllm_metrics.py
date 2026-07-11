from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from locomo_jasper_bench.kv.vllm_metrics import (
    request_timing_from_output,
    require_engine_ttft_ms,
)


def _output(metrics: Any) -> Any:
    return SimpleNamespace(metrics=metrics)


def test_engine_first_token_latency_is_used() -> None:
    assert require_engine_ttft_ms(_output(SimpleNamespace(first_token_latency=0.25))) == 250.0


def test_arrival_and_first_token_times_yield_engine_ttft() -> None:
    timing = request_timing_from_output(
        _output(SimpleNamespace(arrival_time=10.0, first_token_time=10.5))
    )

    assert timing.time_to_first_token_ms == 500.0


def test_missing_engine_metrics_fail_the_run() -> None:
    with pytest.raises(RuntimeError, match="did not report per-request metrics"):
        require_engine_ttft_ms(_output(None))

    with pytest.raises(RuntimeError, match="TTFT cannot be measured"):
        require_engine_ttft_ms(_output(SimpleNamespace()))
