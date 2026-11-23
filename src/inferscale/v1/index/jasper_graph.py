from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import VECTOR_DISTANCE
from ..units import bytes_to_mb

_FIXED_HEADER_BYTES = 4 * 8
_GPU_FLOAT16_ELEMENT_BYTES = 2
_EDGE_COUNT_BYTES = 1
_INT32_EDGE_BYTES = 4


@dataclass(slots=True, frozen=True)
class GraphBuild:
    graph: Any
    vectors_gpu: Any
    memory_stats: dict[str, int | float | None]


def empty_memory_stats() -> dict[str, int | float | None]:
    return {
        "jasper_graph_gpu_bytes": None,
        "jasper_graph_gpu_mb": None,
        "jasper_graph_torch_allocated_delta_bytes": None,
        "jasper_graph_torch_allocated_delta_mb": None,
    }


def build_graph(
    vectors: np.ndarray, *, n_neighbors: int, alpha: float, workspace_budget: str
) -> GraphBuild:
    try:
        import jasper
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Jasper backend requires torch, apache-tvm-ffi, and the built jasper Python package."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Jasper backend requires a CUDA device.")

    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    vectors_gpu = torch.from_numpy(vectors).to(device="cuda", dtype=torch.float16)
    graph = jasper.Graph.build(
        vectors_gpu,
        n_neighbors=n_neighbors,
        distance=VECTOR_DISTANCE,
        alpha=alpha,
        workspace_budget=workspace_budget,
    )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_delta = max(0, int(torch.cuda.memory_allocated()) - allocated_before)

    graph_bytes = serialized_bytes(
        vector_count=int(vectors.shape[0]), dim=int(vectors.shape[1]), n_neighbors=n_neighbors
    )
    return GraphBuild(
        graph=graph,
        vectors_gpu=vectors_gpu,
        memory_stats={
            "jasper_graph_gpu_bytes": graph_bytes,
            "jasper_graph_gpu_mb": bytes_to_mb(graph_bytes),
            "jasper_graph_torch_allocated_delta_bytes": allocated_delta,
            "jasper_graph_torch_allocated_delta_mb": bytes_to_mb(allocated_delta),
        },
    )


def serialized_bytes(*, vector_count: int, dim: int, n_neighbors: int) -> int:
    if vector_count <= 0 or dim <= 0:
        return 0
    float16_vector_bytes = dim * _GPU_FLOAT16_ELEMENT_BYTES
    int32_edges_bytes = n_neighbors * _INT32_EDGE_BYTES
    bytes_per_node = float16_vector_bytes + _EDGE_COUNT_BYTES + int32_edges_bytes
    return _FIXED_HEADER_BYTES + vector_count * bytes_per_node
