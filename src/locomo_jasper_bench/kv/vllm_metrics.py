from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VllmRequestTiming:
    time_to_first_token_ms: float | None = None


def require_engine_ttft_ms(output: Any) -> float:
    """Return the engine-reported TTFT, refusing to run without real metrics.

    TTFT is a headline benchmark metric; fabricating it from wall-clock probes
    or reporting it as missing would silently corrupt cross-run comparisons, so
    an engine build that does not populate RequestOutput.metrics fails the run.
    """
    timing = request_timing_from_output(output)
    if timing.time_to_first_token_ms is None:
        raise RuntimeError(
            "vLLM did not report per-request metrics on RequestOutput.metrics; "
            "TTFT cannot be measured on this engine build. Use a vLLM build that "
            "populates request timing metrics."
        )
    return timing.time_to_first_token_ms


def request_timing_from_output(output: Any) -> VllmRequestTiming:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return VllmRequestTiming()

    first_token_latency = _float_attr(metrics, "first_token_latency")
    if first_token_latency is not None and first_token_latency > 0:
        return VllmRequestTiming(time_to_first_token_ms=first_token_latency * 1000)

    arrival_time = _float_attr(metrics, "arrival_time")
    first_token_time = _float_attr(metrics, "first_token_time")
    if arrival_time is not None and first_token_time is not None and first_token_time >= arrival_time:
        return VllmRequestTiming(time_to_first_token_ms=(first_token_time - arrival_time) * 1000)

    return VllmRequestTiming()


def _float_attr(value: Any, name: str) -> float | None:
    raw = getattr(value, name, None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
