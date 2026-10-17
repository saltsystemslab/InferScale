"""Jasper graph index over chunk embeddings, with exact fallbacks.

Beam search over the graph is the fast path. Requests that need every vector
(full-store top-k, partial payload filters) and beam results that come back
short or padded fall back to exact inner-product scoring, on the GPU when the
fp16 matrix is resident and on the host otherwise, so retrieval is complete
for every caller.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import DeviceSearchResult, SearchHit, SearchMetrics
from ..units import bytes_to_mb
from .filters import payload_matches
from .jasper_graph import build_graph, empty_memory_stats
from .vectors import as_float32_vector

logger = logging.getLogger(__name__)

# Largest effective beam width jasperpy accepts.
MAX_JASPER_BEAM_WIDTH = 959


@dataclass(slots=True)
class JasperIndexConfig:
    n_neighbors: int = 64
    alpha: float = 1.0
    workspace_budget: str = "10GB"
    beam_width: int = 64


class JasperIndex:
    """In-memory vector index with Jasper GPU search.

    Rows are kept in insertion order; a row's position is the stable ID that
    Jasper assigns when the graph is built from the same matrix.
    """

    def __init__(self, config: JasperIndexConfig | None = None) -> None:
        self.config = config if config is not None else JasperIndexConfig()
        self._rows: list[tuple[str, dict[str, Any]]] = []
        self._ordinals: dict[str, int] = {}
        self._vectors: np.ndarray | None = None
        self._vectors_gpu: Any = None
        self._graph: Any = None
        self._finalized = False
        self._graph_memory_stats = empty_memory_stats()

    @property
    def vector_count(self) -> int:
        return len(self._rows)

    @property
    def dim(self) -> int | None:
        return None if self._vectors is None else int(self._vectors.shape[1])

    def count(self, filters: dict[str, Any] | None = None) -> int:
        return sum(1 for _, payload in self._rows if payload_matches(payload, filters))

    def add_many(
        self,
        vectors: Iterable[np.ndarray | list[float]],
        payloads: Iterable[dict[str, Any]],
        ids: Iterable[str] | None = None,
    ) -> list[str]:
        vector_list = [as_float32_vector(vector) for vector in vectors]
        payload_list = [dict(payload) for payload in payloads]
        if len(vector_list) != len(payload_list):
            raise ValueError("vectors and payloads must have the same length")
        if not vector_list:
            return []
        if ids is not None:
            id_list = [str(item_id) for item_id in ids]
        else:
            id_list = [str(uuid.uuid4()) for _ in vector_list]
        if len(id_list) != len(vector_list):
            raise ValueError("ids and vectors must have the same length")

        matrix = np.vstack(vector_list)
        if self._vectors is not None:
            if matrix.shape[1] != self._vectors.shape[1]:
                raise ValueError(
                    f"vector dim {matrix.shape[1]} does not match "
                    f"store dim {self._vectors.shape[1]}"
                )
            matrix = np.vstack([self._vectors, matrix])
        # Any resident graph was built for the previous matrix.
        self.close()
        self._vectors = matrix
        for item_id, payload in zip(id_list, payload_list):
            self._ordinals.setdefault(item_id, len(self._rows))
            self._rows.append((item_id, payload))
        return id_list

    def finalize(self) -> None:
        if self._graph is None and self.vector_count:
            self._graph = self._build_jasper_graph()
        self._finalized = True

    def memory_stats(self) -> dict[str, int | float | None]:
        vector_bytes = int(self._vectors.nbytes) if self._vectors is not None else 0
        gpu_bytes = vector_bytes if self._graph is not None else 0
        return {
            "jasper_vector_count": self.vector_count,
            "jasper_embedding_dim": self.dim,
            "jasper_embedding_matrix_cpu_bytes": vector_bytes,
            "jasper_embedding_matrix_cpu_mb": bytes_to_mb(vector_bytes),
            "jasper_embedding_matrix_gpu_logical_bytes": gpu_bytes,
            "jasper_embedding_matrix_gpu_logical_mb": bytes_to_mb(gpu_bytes),
            **self._graph_memory_stats,
        }

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchHit], SearchMetrics]:
        if not self.vector_count:
            return [], self._metrics(0.0, self.config.beam_width)
        query = self._query(query_vector)
        top_k = max(1, min(top_k, self.vector_count))
        beam_width = max(self.config.beam_width, top_k)
        # A filter that rejects some rows must score every row to stay complete.
        exact_filter = bool(filters) and not self._filter_matches_all(filters)
        candidate_k = self.vector_count if exact_filter else top_k
        hits, search_time_ms = self._search_candidates(query, candidate_k, beam_width)
        if filters:
            hits = [hit for hit in hits if payload_matches(hit.payload, filters)]
        return _ranked(hits[:top_k]), self._metrics(search_time_ms, beam_width)

    def search_device(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> DeviceSearchResult | None:
        """Complete results whose IDs and distances stay on the GPU.

        ``None`` means the SearchHit path is required: before finalization,
        without a resident matrix, or when a payload filter would reject rows.
        """
        if not self.vector_count:
            return None
        query = self._query(query_vector)
        top_k = max(1, min(top_k, self.vector_count))
        if filters and not self._filter_matches_all(filters):
            return None
        if not self._finalized or self._graph is None or self._vectors_gpu is None:
            return None
        beam_width = max(self.config.beam_width, top_k)
        stable_ids, distances, search_time_ms = self._device_candidates(query, top_k, beam_width)
        if not _device_result_is_complete(
            stable_ids, distances, expected_count=top_k, vector_count=self.vector_count
        ):
            raise RuntimeError("Jasper exact GPU search returned incomplete or invalid results.")
        return DeviceSearchResult(
            stable_ids=stable_ids,
            distances=distances,
            metrics=self._metrics(search_time_ms, beam_width),
        )

    def materialize_device_result(self, result: DeviceSearchResult) -> list[SearchHit]:
        """SearchHits for a device result, for bookkeeping outside the KV fast path."""
        return self._hits_for(
            result.stable_ids.detach().cpu().tolist(), result.distances.detach().cpu().tolist()
        )

    def stable_id_items(self) -> tuple[tuple[int, str], ...]:
        """Bindings from Jasper stable IDs (row order) to chunk IDs."""
        if not self._finalized:
            raise RuntimeError("Jasper stable IDs are unavailable before finalization.")
        return tuple((ordinal, item_id) for ordinal, (item_id, _) in enumerate(self._rows))

    def rows(self, filters: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
        return [
            (item_id, dict(payload))
            for item_id, payload in self._rows
            if payload_matches(payload, filters)
        ]

    def get(self, item_id: str) -> SearchHit | None:
        ordinal = self._ordinals.get(str(item_id))
        if ordinal is None:
            return None
        candidate_id, payload = self._rows[ordinal]
        return SearchHit(candidate_id, dict(payload), 1.0, 0.0, ordinal + 1)

    def update(
        self,
        item_id: str,
        *,
        vector: np.ndarray | list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ordinal = self._ordinals.get(str(item_id))
        if ordinal is None:
            raise KeyError(f"Unknown Jasper vector id: {item_id}")
        if payload is None and vector is None:
            return
        next_vector = None
        if vector is not None:
            if self._vectors is None:
                raise RuntimeError("Jasper vector matrix is unavailable.")
            next_vector = as_float32_vector(vector)
            if next_vector.shape[0] != self._vectors.shape[1]:
                raise ValueError(
                    f"vector dim {next_vector.shape[0]} does not match "
                    f"store dim {self._vectors.shape[1]}"
                )
        self.close()
        if payload is not None:
            self._rows[ordinal] = (self._rows[ordinal][0], dict(payload))
        if next_vector is not None:
            self._vectors[ordinal] = next_vector

    def delete(self, item_id: str) -> None:
        ordinal = self._ordinals.get(str(item_id))
        if ordinal is None:
            return
        self.close()
        del self._rows[ordinal]
        if self._rows:
            self._vectors = np.delete(self._vectors, ordinal, axis=0)
        else:
            self._vectors = None
        self._ordinals = {}
        for index, (candidate_id, _) in enumerate(self._rows):
            self._ordinals.setdefault(candidate_id, index)

    def reset(self) -> None:
        self.close()
        self._rows = []
        self._ordinals = {}
        self._vectors = None

    def close(self) -> None:
        if self._graph is not None:
            self._graph.free()
        self._graph = None
        self._vectors_gpu = None
        self._finalized = False
        self._graph_memory_stats = empty_memory_stats()

    def _build_jasper_graph(self) -> Any:
        build = build_graph(
            self._vectors,
            n_neighbors=self.config.n_neighbors,
            alpha=self.config.alpha,
            workspace_budget=self.config.workspace_budget,
        )
        self._vectors_gpu = build.vectors_gpu
        self._graph_memory_stats = build.memory_stats
        return build.graph

    def _search_candidates(
        self, query: np.ndarray, candidate_k: int, beam_width: int
    ) -> tuple[list[SearchHit], float]:
        """Beam search when the graph can answer completely, exact scoring otherwise."""
        beam_ready = self._finalized and self._graph is not None
        # Beam search does not guarantee reaching every vertex, so a request for
        # the whole store must use exact scoring to stay complete.
        if beam_ready and candidate_k < self.vector_count:
            hits, search_time_ms = self._search_jasper(query, candidate_k, beam_width=beam_width)
            if len(hits) >= candidate_k:
                return hits, search_time_ms
            logger.info(
                "Jasper beam search returned %d of %d requested hits; retrying with exact search.",
                len(hits),
                candidate_k,
            )
            exact_hits, exact_time_ms = self._search_exact_gpu(query, candidate_k)
            return exact_hits, search_time_ms + exact_time_ms
        if beam_ready and self._vectors_gpu is not None:
            return self._search_exact_gpu(query, candidate_k)
        return self._search_exact(query, candidate_k)

    def _device_candidates(
        self, query: np.ndarray, top_k: int, beam_width: int
    ) -> tuple[Any, Any, float]:
        if top_k >= self.vector_count:
            return self._search_exact_gpu_device(query, top_k)
        stable_ids, distances, search_time_ms = self._search_jasper_device(
            query, top_k, beam_width=beam_width
        )
        if _device_result_is_complete(
            stable_ids, distances, expected_count=top_k, vector_count=self.vector_count
        ):
            return stable_ids, distances, search_time_ms
        logger.info(
            "Jasper device search returned incomplete or invalid results; "
            "retrying with exact GPU search."
        )
        stable_ids, distances, exact_time_ms = self._search_exact_gpu_device(query, top_k)
        return stable_ids, distances, search_time_ms + exact_time_ms

    def _search_jasper(
        self, query: np.ndarray, top_k: int, *, beam_width: int
    ) -> tuple[list[SearchHit], float]:
        ordinals, distances, search_time_ms = self._search_jasper_device(
            query, top_k, beam_width=beam_width
        )
        ordinal_values = [int(ordinal) for ordinal in ordinals.detach().cpu().numpy()]
        distance_values = list(distances.detach().cpu().numpy())
        # Jasper pads the row with -1 when fewer than k vertices were reached.
        found = next(
            (index for index, ordinal in enumerate(ordinal_values) if ordinal < 0),
            len(ordinal_values),
        )
        return self._hits_for(ordinal_values[:found], distance_values[:found]), search_time_ms

    def _search_jasper_device(
        self, query: np.ndarray, top_k: int, *, beam_width: int
    ) -> tuple[Any, Any, float]:
        if self._graph is None:
            raise RuntimeError("Jasper graph must be finalized before GPU search.")
        import torch

        query_tensor = torch.from_numpy(query.reshape(1, -1)).to(device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        started = time.perf_counter()
        indices, distances = self._graph.search(query_tensor, k=top_k, beam_width=beam_width)
        torch.cuda.synchronize()
        search_time_ms = (time.perf_counter() - started) * 1000
        return indices[0].detach(), distances[0].detach(), search_time_ms

    def _search_exact_gpu(self, query: np.ndarray, top_k: int) -> tuple[list[SearchHit], float]:
        if self._vectors_gpu is None:
            return self._search_exact(query, top_k)
        ordinals, distances, search_time_ms = self._search_exact_gpu_device(query, top_k)
        hits = self._hits_for(ordinals.detach().cpu().tolist(), distances.detach().cpu().tolist())
        return hits, search_time_ms

    def _search_exact_gpu_device(self, query: np.ndarray, top_k: int) -> tuple[Any, Any, float]:
        import torch

        matrix = self._vectors_gpu
        if matrix is None:
            raise RuntimeError("Jasper exact GPU search requires a resident vector matrix.")
        query_tensor = torch.from_numpy(query).to(device=matrix.device, dtype=torch.float32)
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = matrix.float() @ query_tensor
        top_scores, top_ordinals = torch.topk(scores, min(int(top_k), int(scores.shape[0])))
        torch.cuda.synchronize()
        search_time_ms = (time.perf_counter() - started) * 1000
        return top_ordinals.detach(), (-top_scores).detach(), search_time_ms

    def _search_exact(self, query: np.ndarray, top_k: int) -> tuple[list[SearchHit], float]:
        started = time.perf_counter()
        if self._vectors is None:
            return [], 0.0
        scores = self._vectors @ query
        ordinals = np.argsort(-scores, kind="stable")[:top_k]
        hits = self._hits_for(ordinals.tolist(), (-scores[ordinals]).tolist())
        return hits, (time.perf_counter() - started) * 1000

    def _hits_for(self, ordinals: Iterable[Any], distances: Iterable[Any]) -> list[SearchHit]:
        """Hits in the given order; ordinals index the rows of this index."""
        hits: list[SearchHit] = []
        for ordinal, raw_distance in zip(ordinals, distances):
            ordinal = int(ordinal)
            if not 0 <= ordinal < len(self._rows):
                raise RuntimeError(f"Jasper returned invalid vector ordinal {ordinal}.")
            item_id, payload = self._rows[ordinal]
            distance = float(raw_distance)
            hits.append(SearchHit(item_id, dict(payload), -distance, distance, len(hits) + 1))
        return hits

    def _query(self, query_vector: np.ndarray | list[float]) -> np.ndarray:
        query = as_float32_vector(query_vector, name="query_vector")
        if query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self.dim}")
        return query

    def _filter_matches_all(self, filters: dict[str, Any]) -> bool:
        return all(payload_matches(payload, filters) for _, payload in self._rows)

    def _metrics(self, search_time_ms: float, beam_width: int) -> SearchMetrics:
        return SearchMetrics(
            search_time_ms, vector_backend="jasper", jasper_effective_beam_width=beam_width
        )


def _ranked(hits: list[SearchHit]) -> list[SearchHit]:
    for rank, hit in enumerate(hits, start=1):
        hit.rank = rank
    return hits


def _device_result_is_complete(
    stable_ids: Any, distances: Any, *, expected_count: int, vector_count: int
) -> bool:
    """True when every row is a distinct in-range ID with a finite distance."""
    import torch

    if int(stable_ids.numel()) != expected_count or int(distances.numel()) != expected_count:
        return False
    valid = (
        (stable_ids >= 0)
        & (stable_ids < vector_count)
        & torch.isfinite(distances)
        & (distances.abs() < 1e30)
    ).all()
    if expected_count > 1:
        sorted_ids = torch.sort(stable_ids).values
        valid = valid & (sorted_ids[1:] != sorted_ids[:-1]).all()
    return bool(valid.item())
