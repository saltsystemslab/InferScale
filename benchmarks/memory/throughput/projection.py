from __future__ import annotations

from dataclasses import dataclass

from benchmarks.common.system import available_host_memory_bytes
from benchmarks.memory.throughput.config import ThroughputConfig

GIB = 2**30
JASPER_SEGMENT_SLOTS = 1 << 12
MEASURED_BYTES_PER_SLOT = 3457
JASPER_GRAPH_DEVICE_BYTES = JASPER_SEGMENT_SLOTS * MEASURED_BYTES_PER_SLOT
USABLE_DEVICE_FRACTION = 0.97
PINNABLE_HOST_FRACTION = 0.9


@dataclass(frozen=True, slots=True)
class CorpusFootprint:
    fact_count: int
    token_count: int
    kv_bytes: int
    scaffold_token_count: int

    @property
    def bytes_per_token(self) -> float:
        return self.kv_bytes / self.token_count if self.token_count else 0.0


def check_kv_gpu_projection(
    config: ThroughputConfig,
    *,
    num_users: int,
    total_requests: int,
    unique_sample_count: int,
    footprint: CorpusFootprint,
) -> None:
    if footprint.fact_count == 0 or footprint.token_count == 0:
        return
    retrieved_facts = min(config.top_k, footprint.fact_count)
    average_fact_tokens = footprint.token_count / footprint.fact_count
    composed_tokens = footprint.scaffold_token_count + retrieved_facts * average_fact_tokens
    composed_bytes = total_requests * composed_tokens * footprint.bytes_per_token
    source_bytes = footprint.kv_bytes * unique_sample_count
    if config.kv_store_backend == "cpu":
        corpus_resident_bytes = composed_tokens * footprint.bytes_per_token
    else:
        corpus_resident_bytes = source_bytes
    graph_bytes = unique_sample_count * JASPER_GRAPH_DEVICE_BYTES
    device_total = _device_total_memory_bytes()
    vllm_pool = config.kv_gpu_memory_utilization * device_total
    phase_peaks = {
        "setup": graph_bytes + composed_bytes + corpus_resident_bytes,
        "generate": vllm_pool + composed_bytes + graph_bytes,
    }
    for phase, projected_peak in phase_peaks.items():
        if projected_peak > USABLE_DEVICE_FRACTION * device_total:
            raise RuntimeError(
                f"Projected KV GPU footprint exceeds device memory in the {phase} phase: "
                f"projected={_gib(projected_peak):.1f}GiB vllm_pool={_gib(vllm_pool):.1f}GiB "
                f"composed={_gib(composed_bytes):.1f}GiB sources={_gib(source_bytes):.1f}GiB "
                f"graphs={_gib(graph_bytes):.1f}GiB device={_gib(device_total):.1f}GiB "
                f"(users={num_users}, requests={total_requests}). Reduce user_counts, top_k, "
                "or inferscale.engine.gpu_memory_utilization in the run JSON."
            )
    if config.kv_store_backend == "cpu":
        check_pinned_host_projection(
            config, corpus_bytes=source_bytes, num_users=num_users, total_requests=total_requests
        )


def check_pinned_host_projection(
    config: ThroughputConfig, *, corpus_bytes: float, num_users: int, total_requests: int
) -> None:
    del config
    available_bytes = available_host_memory_bytes()
    if available_bytes is None:
        return
    if corpus_bytes > PINNABLE_HOST_FRACTION * available_bytes:
        raise RuntimeError(
            "Projected pinned-host KV corpus exceeds available RAM: "
            f"corpus={_gib(corpus_bytes):.1f}GiB available={_gib(available_bytes):.1f}GiB "
            f"(users={num_users}, requests={total_requests}). "
            "Reduce the corpus or free host memory."
        )


def _device_total_memory_bytes() -> int:
    import torch

    return int(torch.cuda.get_device_properties(0).total_memory)


def _gib(byte_count: float) -> float:
    return byte_count / GIB
