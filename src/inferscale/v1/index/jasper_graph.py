"""Jasper graph construction and the GPU memory accounting around it."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import VECTOR_DISTANCE
from ..units import bytes_to_mb

# jasperpy serializes a graph as a fixed header followed by one node per
# vector: the float16 vector, one edge-count byte, and n_neighbors int32 edges.
_FILE_HEADER_BYTES = 4 * 8
_EDGE_INDEX_BYTES = 4
_EDGE_COUNT_BYTES = 1
# jasperpy keeps vectors on the GPU as float16 whatever the input dtype.
_VECTOR_ELEMENT_BYTES = 2


@dataclass(slots=True, frozen=True)
class GraphBuild:
    """A built graph, the fp16 matrix it searches, and the memory it took."""

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
    """Build the graph on CUDA and keep the fp16 matrix resident next to it.

    jasper has no brute-force API, so the matrix stays on the GPU for the
    exact searches (full-store requests and partial-filter fallbacks); the
    allocated-delta statistic includes it.
    """
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
    """Size of the graph in jasperpy's serialized layout."""
    if vector_count <= 0 or dim <= 0:
        return 0
    bytes_per_node = (
        dim * _VECTOR_ELEMENT_BYTES + _EDGE_COUNT_BYTES + n_neighbors * _EDGE_INDEX_BYTES
    )
    return _FILE_HEADER_BYTES + vector_count * bytes_per_node
