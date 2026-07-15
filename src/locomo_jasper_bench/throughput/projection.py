"""Pre-flight GPU and host memory projections for the kv_injection condition."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import ThroughputConfig

# Device bytes one jasper sample graph allocates: one segment of
# 1u << 12 slots x ~3,457 bytes/slot on the smaller-vectors-per-segment
# jasperpy branch. Jasper sizes segments internally, so this is an
# estimate tied to that branch, not a bound.
JASPER_GRAPH_DEVICE_BYTES = 4096 * 3457


def check_kv_gpu_projection(
    config: ThroughputConfig,
    *,
    num_users: int,
    total_requests: int,
    unique_sample_count: int,
    scaffold_chunks: tuple[Any, Any],
    first_sample_chunks: dict[str, Any],
) -> None:
    """Fail before the expensive precompute if the KV footprint cannot fit.

    kv_injection runs setup (retrieve + compose) fully before the engine
    starts and then drops the source chunks, so the peaks to check are:
    setup holds source chunks + jasper graphs + composed copies, and
    generation holds the vLLM pool + the backend's resident KV + the
    still-open jasper graphs.
    """
    import torch

    from ..kv.gpu_memory_store import _tensor_nbytes

    def chunk_bytes(chunk: Any) -> int:
        return sum(_tensor_nbytes(tensor) for tensor in chunk.kv_by_layer.values())

    first_sample_bytes = sum(chunk_bytes(chunk) for chunk in first_sample_chunks.values())
    first_sample_tokens = sum(len(chunk.token_ids) for chunk in first_sample_chunks.values())
    if first_sample_tokens <= 0 or not first_sample_chunks:
        return
    bytes_per_token = first_sample_bytes / first_sample_tokens

    header_chunk, footer_chunk = scaffold_chunks
    scaffold_tokens = len(header_chunk.token_ids) + len(footer_chunk.token_ids)
    retrieved_facts = min(config.top_k, len(first_sample_chunks))
    average_fact_tokens = first_sample_tokens / len(first_sample_chunks)
    composed_tokens = scaffold_tokens + retrieved_facts * average_fact_tokens
    composed_bytes = total_requests * composed_tokens * bytes_per_token
    source_bytes = first_sample_bytes * unique_sample_count

    device_total = torch.cuda.get_device_properties(0).total_memory
    vllm_pool = config.kv_gpu_memory_utilization * device_total

    # Per-backend HBM-resident component: cpu holds composed memories
    # in pinned host RAM (one transient copy while composing, one staged
    # copy per slot while generating); the GPU store keeps every composed
    # copy resident from composition until generation ends.
    if config.kv_store_backend == "cpu":
        compose_resident_bytes = composed_tokens * bytes_per_token
        generate_resident_bytes = config.kv_staging_slots * composed_tokens * bytes_per_token
        remediation = "Shrink the user count, lower --top-k, --kv-staging-slots, or --gpu-memory-utilization."
    else:
        compose_resident_bytes = composed_bytes
        generate_resident_bytes = composed_bytes
        remediation = "Shrink the user count, lower --top-k, or reduce --gpu-memory-utilization."

    graph_bytes = unique_sample_count * JASPER_GRAPH_DEVICE_BYTES
    phase_peaks = {
        "setup": source_bytes + graph_bytes + compose_resident_bytes,
        "generate": vllm_pool + generate_resident_bytes + graph_bytes,
    }
    for phase, projected_peak in phase_peaks.items():
        if projected_peak > 0.97 * device_total:
            raise RuntimeError(
                f"Projected KV GPU footprint exceeds device memory in the {phase} phase: "
                f"projected={projected_peak / 2**30:.1f}GiB "
                f"vllm_pool={vllm_pool / 2**30:.1f}GiB "
                f"composed={composed_bytes / 2**30:.1f}GiB "
                f"sources={source_bytes / 2**30:.1f}GiB "
                f"graphs={graph_bytes / 2**30:.1f}GiB "
                f"device={device_total / 2**30:.1f}GiB "
                f"(users={num_users}, requests={total_requests}). "
                f"{remediation}"
            )

    if config.kv_store_backend == "cpu":
        check_pinned_host_projection(
            config,
            composed_bytes=composed_bytes,
            num_users=num_users,
            total_requests=total_requests,
        )


def parse_mem_available_bytes(meminfo_text: str) -> int | None:
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def available_host_memory_bytes() -> int | None:
    try:
        available = parse_mem_available_bytes(
            Path("/proc/meminfo").read_text(encoding="ascii")
        )
    except OSError:
        available = None
    if available is not None:
        return available
    # Without MemAvailable, treat 80% of physical RAM as available; the
    # kernel, page cache, and resident processes own an unknown share.
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.8)
    except (ValueError, OSError, AttributeError):
        return None


def check_pinned_host_projection(
    config: ThroughputConfig,
    *,
    composed_bytes: float,
    num_users: int,
    total_requests: int,
) -> None:
    """Fail before precompute if the pinned KV pool cannot fit in host RAM.

    Pinned allocations are non-swappable, so the ceiling is memory that is
    actually available now, with headroom for the vLLM engine's own host
    allocations after this check.
    """
    available = available_host_memory_bytes()
    if available is None:
        return
    if composed_bytes > 0.9 * available:
        raise RuntimeError(
            "Projected pinned-host KV footprint exceeds available RAM: "
            f"composed={composed_bytes / 2**30:.1f}GiB "
            f"available={available / 2**30:.1f}GiB "
            f"(users={num_users}, requests={total_requests}). "
            "Shrink the user count, lower --top-k, or free host memory."
        )
