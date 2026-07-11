from __future__ import annotations

import gc
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..vector_types import SearchHit, SearchMetrics, VECTOR_DISTANCE, VectorStoreConfig
from .store_utils import payload_matches

_JASPER_GRAPH_FILE_HEADER_BYTES = 4 * 8
_JASPER_INDEX_BYTES = 4
_JASPER_EDGE_COUNT_BYTES = 1


class JasperVectorStore:
    """In-memory vector store with Jasper GPU search."""

    def __init__(self, root: str | Path, config: VectorStoreConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self._payloads_by_ordinal: dict[int, tuple[str, dict[str, Any]]] = {}
        self._vectors: np.ndarray | None = None
        self._graph: Any = None
        self._finalized = False
        self._jasper_graph_memory_stats: dict[str, int | float | None] = _empty_graph_memory_stats()

    @property
    def vector_count(self) -> int:
        if self._vectors is None:
            return 0
        return int(self._vectors.shape[0])

    @property
    def dim(self) -> int | None:
        if self._vectors is None:
            return None
        return int(self._vectors.shape[1])

    def count(self, filters: dict[str, Any] | None = None) -> int:
        return sum(
            1
            for _, payload in self._payloads_by_ordinal.values()
            if payload_matches(payload, filters)
        )

    def add_many(
        self,
        vectors: Iterable[np.ndarray | list[float]],
        payloads: Iterable[dict[str, Any]],
        ids: Iterable[str] | None = None,
    ) -> list[str]:
        vector_list = [_as_float32_vector(vector) for vector in vectors]
        payload_list = list(payloads)
        if len(vector_list) != len(payload_list):
            raise ValueError("vectors and payloads must have the same length")
        if not vector_list:
            return []

        id_list = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in vector_list]
        if len(id_list) != len(vector_list):
            raise ValueError("ids and vectors must have the same length")
        if self._graph is not None:
            self.close()

        matrix = np.vstack(vector_list)
        if self._vectors is None:
            self._vectors = matrix
            start_ord = 0
        else:
            if matrix.shape[1] != self._vectors.shape[1]:
                raise ValueError(f"vector dim {matrix.shape[1]} does not match store dim {self._vectors.shape[1]}")
            start_ord = int(self._vectors.shape[0])
            self._vectors = np.vstack([self._vectors, matrix])

        for offset, (item_id, payload) in enumerate(zip(id_list, payload_list)):
            item_id = str(item_id)
            ordinal = start_ord + offset
            self._payloads_by_ordinal[ordinal] = (item_id, dict(payload))
        self._graph = None
        self._finalized = False
        self._jasper_graph_memory_stats = _empty_graph_memory_stats()
        return id_list

    def finalize(self) -> None:
        if self._vectors is None or self._vectors.size == 0:
            self._finalized = True
            return
        if self._graph is None:
            self._graph = self._build_jasper_graph()
        self._finalized = True

    def memory_stats(self) -> dict[str, int | float | None]:
        vector_bytes = int(self._vectors.nbytes) if self._vectors is not None else 0
        graph_loaded = self._graph is not None
        logical_gpu_bytes = vector_bytes if graph_loaded else 0
        return {
            "jasper_vector_count": self.vector_count,
            "jasper_embedding_dim": self.dim,
            "jasper_embedding_matrix_cpu_bytes": vector_bytes,
            "jasper_embedding_matrix_cpu_mb": _bytes_to_mb(vector_bytes),
            "jasper_embedding_matrix_gpu_logical_bytes": logical_gpu_bytes,
            "jasper_embedding_matrix_gpu_logical_mb": _bytes_to_mb(logical_gpu_bytes),
            **self._jasper_graph_memory_stats,
        }

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchHit], SearchMetrics]:
        if self._vectors is None or self._vectors.size == 0:
            return [], SearchMetrics(
                0.0,
                vector_backend="jasper",
                jasper_effective_beam_width=self.config.beam_width,
            )
        query = _as_float32_vector(query_vector, name="query_vector")
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self._vectors.shape[1]}")
        top_k = max(1, min(top_k, self.vector_count))
        filters_match_all = bool(filters) and all(
            payload_matches(payload, filters)
            for _, payload in self._payloads_by_ordinal.values()
        )
        requires_exact_filtering = bool(filters) and not filters_match_all
        candidate_k = self.vector_count if requires_exact_filtering else top_k
        effective_beam_width = max(self.config.beam_width, top_k)
        if self._finalized and self._graph is not None and not requires_exact_filtering:
            hits, search_time_ms = self._search_jasper(
                query,
                candidate_k,
                beam_width=effective_beam_width,
            )
        else:
            hits, search_time_ms = self._search_exact(query, candidate_k)
        if filters:
            hits = [hit for hit in hits if payload_matches(hit.payload, filters)]
        hits = hits[:top_k]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits, SearchMetrics(
            search_time_ms,
            vector_backend="jasper",
            jasper_effective_beam_width=effective_beam_width,
        )

    def rows(self, filters: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
        return [
            (item_id, dict(payload))
            for ordinal in sorted(self._payloads_by_ordinal)
            for item_id, payload in [self._payloads_by_ordinal[ordinal]]
            if payload_matches(payload, filters)
        ]

    def get(self, item_id: str) -> SearchHit | None:
        ordinal = self._ordinal_for_id(item_id)
        if ordinal is None:
            return None
        candidate_id, payload = self._payloads_by_ordinal[ordinal]
        return SearchHit(candidate_id, dict(payload), 1.0, 0.0, ordinal + 1)

    def update(
        self,
        item_id: str,
        *,
        vector: np.ndarray | list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ordinal = self._ordinal_for_id(item_id)
        if ordinal is None:
            raise KeyError(f"Unknown Jasper vector id: {item_id}")
        current_id, _ = self._payloads_by_ordinal[ordinal]
        if payload is not None:
            self._payloads_by_ordinal[ordinal] = (current_id, dict(payload))
        if vector is not None:
            if self._vectors is None:
                raise RuntimeError("Jasper vector matrix is unavailable.")
            next_vector = _as_float32_vector(vector)
            if next_vector.shape[0] != self._vectors.shape[1]:
                raise ValueError(
                    f"vector dim {next_vector.shape[0]} does not match store dim {self._vectors.shape[1]}"
                )
            self._vectors[ordinal] = next_vector
        if payload is not None or vector is not None:
            self.close()

    def delete(self, item_id: str) -> None:
        ordinal = self._ordinal_for_id(item_id)
        if ordinal is None:
            return
        rows = [
            (candidate_id, payload, self._vectors[index] if self._vectors is not None else None)
            for index, (candidate_id, payload) in sorted(self._payloads_by_ordinal.items())
            if index != ordinal
        ]
        self.close()
        self._payloads_by_ordinal = {
            index: (candidate_id, dict(payload))
            for index, (candidate_id, payload, _) in enumerate(rows)
        }
        self._vectors = (
            np.vstack([vector for _, _, vector in rows]).astype(np.float32, copy=False)
            if rows
            else None
        )

    def reset(self) -> None:
        self.close()
        self._payloads_by_ordinal.clear()
        self._vectors = None

    def close(self) -> None:
        if self._graph is not None:
            free = getattr(self._graph, "free", None)
            if callable(free):
                free()
        self._graph = None
        self._finalized = False
        self._jasper_graph_memory_stats = _empty_graph_memory_stats()

    def _build_jasper_graph(self) -> Any:
        try:
            import torch
            import jasper
        except ImportError as exc:
            raise RuntimeError(
                "Jasper backend requires torch, apache-tvm-ffi, and the built jasper Python package. "
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Jasper backend requires a CUDA device.")
        _cuda_synchronize(torch)
        allocated_before = _cuda_memory_allocated(torch)

        vectors = torch.from_numpy(self._vectors).to(device="cuda", dtype=torch.float32)
        graph = jasper.Graph.build(
            vectors,
            n_neighbors=self.config.n_neighbors,
            distance=VECTOR_DISTANCE,
            alpha=self.config.alpha,
            workspace_budget=self.config.workspace_budget,
        )

        _cuda_synchronize(torch)
        del vectors
        gc.collect()
        empty_cache = getattr(torch.cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        _cuda_synchronize(torch)
        allocated_after = _cuda_memory_allocated(torch)

        graph_gpu_bytes = _jasper_graph_serialized_bytes(
            vector_count=self.vector_count,
            dim=int(self._vectors.shape[1]),
            n_neighbors=self.config.n_neighbors,
            data_type_bytes=int(self._vectors.dtype.itemsize),
        )
        torch_allocated_delta_bytes = _positive_delta(allocated_before, allocated_after)
        self._jasper_graph_memory_stats = {
            "jasper_graph_gpu_bytes": graph_gpu_bytes,
            "jasper_graph_gpu_mb": _bytes_to_mb(graph_gpu_bytes),
            "jasper_graph_torch_allocated_delta_bytes": torch_allocated_delta_bytes,
            "jasper_graph_torch_allocated_delta_mb": _bytes_to_mb(torch_allocated_delta_bytes),
        }
        return graph

    def _search_jasper(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        beam_width: int,
    ) -> tuple[list[SearchHit], float]:
        if self._graph is None:
            raise RuntimeError("Jasper graph must be finalized before GPU search.")
        import torch

        query_tensor = torch.from_numpy(query.reshape(1, -1)).to(device="cuda", dtype=torch.float32)
        synchronize = getattr(getattr(torch, "cuda", None), "synchronize", None)
        if callable(synchronize):
            synchronize()
        started = time.perf_counter()
        indices, distances = self._graph.search(query_tensor, k=top_k, beam_width=beam_width)
        if callable(synchronize):
            synchronize()
        search_time_ms = (time.perf_counter() - started) * 1000

        index_values = indices[0].detach().cpu().numpy()
        distance_values = distances[0].detach().cpu().numpy()
        hits: list[SearchHit] = []
        for ordinal, raw_distance in zip(index_values, distance_values):
            row = self._payload_by_ordinal(int(ordinal))
            if row is None:
                raise RuntimeError(f"Jasper returned invalid vector ordinal {int(ordinal)}.")
            item_id, payload = row
            distance = float(raw_distance)
            score = -distance
            hits.append(
                SearchHit(
                    id=item_id,
                    payload=payload,
                    score=score,
                    distance=distance,
                    rank=len(hits) + 1,
                )
            )
        return hits, search_time_ms

    def _search_exact(self, query: np.ndarray, top_k: int) -> tuple[list[SearchHit], float]:
        started = time.perf_counter()
        if self._vectors is None:
            return [], 0.0
        scores = self._vectors @ query
        ordinals = np.argsort(-scores, kind="stable")[:top_k]
        hits: list[SearchHit] = []
        for ordinal in ordinals:
            row = self._payload_by_ordinal(int(ordinal))
            if row is None:
                continue
            item_id, payload = row
            score = float(scores[int(ordinal)])
            hits.append(
                SearchHit(
                    id=item_id,
                    payload=dict(payload),
                    score=score,
                    distance=-score,
                    rank=len(hits) + 1,
                )
            )
        return hits, (time.perf_counter() - started) * 1000

    def _ordinal_for_id(self, item_id: str) -> int | None:
        requested = str(item_id)
        for ordinal, (candidate_id, _) in self._payloads_by_ordinal.items():
            if candidate_id == requested:
                return ordinal
        return None

    def _payload_by_ordinal(self, ordinal: int) -> tuple[str, dict[str, Any]] | None:
        return self._payloads_by_ordinal.get(ordinal)


def _as_float32_vector(vector: np.ndarray | list[float], *, name: str = "vector") -> np.ndarray:
    array = _as_float32_array(vector)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not array.flags.c_contiguous:
        return np.ascontiguousarray(array, dtype=np.float32)
    return array


def _as_float32_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == np.float32 and value.flags.c_contiguous:
        return value
    return np.ascontiguousarray(value, dtype=np.float32)


def _empty_graph_memory_stats() -> dict[str, int | float | None]:
    return {
        "jasper_graph_gpu_bytes": None,
        "jasper_graph_gpu_mb": None,
        "jasper_graph_torch_allocated_delta_bytes": None,
        "jasper_graph_torch_allocated_delta_mb": None,
    }


def _jasper_graph_serialized_bytes(
    *,
    vector_count: int,
    dim: int,
    n_neighbors: int,
    data_type_bytes: int,
) -> int:
    if vector_count <= 0 or dim <= 0:
        return 0

    bytes_per_node = (
        int(dim) * int(data_type_bytes)
        + _JASPER_EDGE_COUNT_BYTES
        + int(n_neighbors) * _JASPER_INDEX_BYTES
    )
    return _JASPER_GRAPH_FILE_HEADER_BYTES + int(vector_count) * bytes_per_node


def _cuda_synchronize(torch: Any) -> None:
    synchronize = getattr(getattr(torch, "cuda", None), "synchronize", None)
    if callable(synchronize):
        synchronize()


def _cuda_memory_allocated(torch: Any) -> int | None:
    memory_allocated = getattr(getattr(torch, "cuda", None), "memory_allocated", None)
    if not callable(memory_allocated):
        return None
    try:
        return int(memory_allocated())
    except Exception:
        return None


def _positive_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return max(0, int(after) - int(before))


def _bytes_to_mb(byte_count: int | None) -> float | None:
    if byte_count is None:
        return None
    return int(byte_count) / (1024 * 1024)
