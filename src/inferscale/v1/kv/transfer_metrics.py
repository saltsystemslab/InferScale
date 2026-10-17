"""Host-to-device transfer records of the pinned store and their summaries."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class TransferRecord:
    """One memory's H2D staging transfer.

    ``staging_stall_ms`` is the time consumers blocked waiting for layer
    copies, the transfer's critical-path exposure; the rest of
    ``h2d_latency_ms`` overlapped with compute.
    """

    user_id: str
    num_tokens: int
    num_bytes: int
    h2d_latency_ms: float
    staging_stall_ms: float


def overlap_ratio(record: TransferRecord) -> float:
    """Fraction of the transfer hidden behind compute, clamped to [0, 1]."""
    if record.h2d_latency_ms <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - record.staging_stall_ms / record.h2d_latency_ms))


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile of an ascending list; 0 when empty."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class TransferMetrics:
    """Transfer records of one store; totals are derived so they cannot drift."""

    records: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, record: TransferRecord) -> None:
        with self.lock:
            self.records.append(record)

    def totals(self) -> dict[str, float]:
        """Running totals for before-and-after attribution to one request."""
        with self.lock:
            records = list(self.records)
        return {
            "total_transfers": float(len(records)),
            "total_bytes_transferred": float(sum(record.num_bytes for record in records)),
            "total_h2d_latency_ms": sum(record.h2d_latency_ms for record in records),
            "total_staging_stall_ms": sum(record.staging_stall_ms for record in records),
        }

    def summary(self) -> dict[str, float | int]:
        with self.lock:
            records = list(self.records)
        latencies = sorted(record.h2d_latency_ms for record in records)
        stalls = sorted(record.staging_stall_ms for record in records)
        timed = [record for record in records if record.h2d_latency_ms > 0]
        overlaps = [overlap_ratio(record) for record in timed]
        bandwidths = [(record.num_bytes / 1e9) / (record.h2d_latency_ms / 1e3) for record in timed]
        return {
            "total_transfers": len(records),
            "total_bytes_transferred": sum(record.num_bytes for record in records),
            "avg_h2d_latency_ms": _mean(latencies),
            "p50_h2d_latency_ms": percentile(latencies, 0.50),
            "p95_h2d_latency_ms": percentile(latencies, 0.95),
            "p99_h2d_latency_ms": percentile(latencies, 0.99),
            "avg_overlap_ratio": _mean(overlaps),
            "avg_effective_bandwidth_gb_per_s": _mean(bandwidths),
            "total_staging_stall_ms": sum(stalls),
            "avg_staging_stall_ms": _mean(stalls),
            "p95_staging_stall_ms": percentile(stalls, 0.95),
        }
