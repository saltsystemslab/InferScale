from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .vector_types import SearchHit, VectorStoreConfig


class JasperVectorStore:
    """In-memory vector store with Jasper GPU search."""

    def __init__(self, root: str | Path, config: VectorStoreConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self._payloads_by_ordinal: dict[int, tuple[str, dict[str, Any]]] = {}
        self._vectors: np.ndarray | None = None
        self._graph: Any = None

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

    def add_many(
        self,
        vectors: Iterable[np.ndarray | list[float]],
        payloads: Iterable[dict[str, Any]],
        ids: Iterable[str] | None = None,
    ) -> list[str]:
        vector_list = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        payload_list = list(payloads)
        if len(vector_list) != len(payload_list):
            raise ValueError("vectors and payloads must have the same length")
        if not vector_list:
            return []

        id_list = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in vector_list]
        if len(id_list) != len(vector_list):
            raise ValueError("ids and vectors must have the same length")

        matrix = np.vstack(vector_list).astype(np.float32, copy=False)
        if self._vectors is None:
            self._vectors = matrix
            start_ord = 0
        else:
            if matrix.shape[1] != self._vectors.shape[1]:
                raise ValueError(f"vector dim {matrix.shape[1]} does not match store dim {self._vectors.shape[1]}")
            start_ord = int(self._vectors.shape[0])
            self._vectors = np.vstack([self._vectors, matrix]).astype(np.float32, copy=False)

        for offset, (item_id, payload) in enumerate(zip(id_list, payload_list)):
            item_id = str(item_id)
            ordinal = start_ord + offset
            self._payloads_by_ordinal[ordinal] = (item_id, dict(payload))
        self._graph = None
        return id_list

    def finalize(self) -> None:
        if self._vectors is None or self._vectors.size == 0:
            return

        self._graph = self._build_jasper_graph()

    def search(self, query_vector: np.ndarray | list[float], top_k: int) -> list[SearchHit]:
        if self._vectors is None or self._vectors.size == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self._vectors.shape[1]}")
        top_k = max(1, min(top_k, self.vector_count))

        return self._search_jasper(query, top_k)

    def close(self) -> None:
        if self._graph is not None:
            free = getattr(self._graph, "free", None)
            if callable(free):
                free()
            self._graph = None

    def _build_jasper_graph(self) -> Any:
        try:
            import torch
            import jasper
        except ImportError as exc:
            raise RuntimeError(
                "Jasper backend requires torch, apache-tvm-ffi, and the built jasper Python package. "
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Jasper backend requires a CUDA device."
            )
        vectors = torch.from_numpy(self._vectors).to(device="cuda", dtype=torch.float32)

        return jasper.Graph.build(
            vectors,
            n_neighbors=self.config.n_neighbors,
            distance=self.config.distance,
            alpha=self.config.alpha,
            workspace_budget=self.config.workspace_budget,
        )

    def _search_jasper(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        if self._graph is None:
            self._graph = self._build_jasper_graph()
        import torch

        query_tensor = torch.from_numpy(query.reshape(1, -1)).to(device="cuda", dtype=torch.float32)
        indices, distances = self._graph.search(query_tensor, k=top_k, beam_width=self.config.beam_width)

        index_values = indices[0].detach().cpu().numpy().astype(np.int64)
        distance_values = distances[0].detach().cpu().numpy().astype(np.float32)
        hits: list[SearchHit] = []
        for ordinal, distance in zip(index_values.tolist(), distance_values.tolist()):
            row = self._payload_by_ordinal(int(ordinal))
            if row is None:
                continue
            item_id, payload = row
            score = float(-distance)
            hits.append(
                SearchHit(
                    id=item_id,
                    payload=payload,
                    score=score,
                    distance=float(distance),
                    rank=len(hits) + 1,
                )
            )
        return hits

    def _payload_by_ordinal(self, ordinal: int) -> tuple[str, dict[str, Any]] | None:
        return self._payloads_by_ordinal.get(ordinal)
