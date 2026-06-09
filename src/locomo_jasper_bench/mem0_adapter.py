from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .jasper_vector_store import JasperVectorStore
from .qdrant_vector_store import QdrantVectorStore
from .vector_types import SearchHit, SearchMetrics, VectorStoreConfig

_MIRRORED_METADATA_KEYS = (
    "user_id",
    "sample_id",
    "turn_id",
    "dia_id",
    "session_id",
    "turn_index",
    "speaker",
    "timestamp",
    "role",
)

try:
    from mem0.vector_stores.base import VectorStoreBase
except Exception:  # pragma: no cover - mem0 is optional for local unit tests

    class VectorStoreBase:  # type: ignore[no-redef]
        pass


class Mem0JasperVectorStore(VectorStoreBase):
    """Mem0 VectorStoreBase adapter backed by the local JasperVectorStore."""

    def __init__(
        self,
        *,
        collection_name: str = "memories",
        embedding_model_dims: int | None = 1536,
        path: str = "/tmp/jasper",
        backend: str = "jasper",
        distance: str = "ip",
        n_neighbors: int = 64,
        alpha: float = 1.0,
        workspace_budget: str = "10GB",
        beam_width: int = 64,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.root = Path(path) / collection_name
        self.config = VectorStoreConfig(
            backend=backend,
            distance=distance,
            n_neighbors=n_neighbors,
            alpha=alpha,
            workspace_budget=workspace_budget,
            beam_width=beam_width,
        )
        self.store = self._create_store()
        self.last_search_metrics = SearchMetrics(search_time_ms=0.0)

    def create_col(self, name: str | None = None, vector_size: int | None = None, distance: str | None = None) -> None:
        if name and name != self.collection_name:
            close = getattr(self.store, "close", None)
            if callable(close):
                close()
            self.collection_name = name
            self.root = self.root.parent / name
            self.store = self._create_store()
        if vector_size is not None:
            self.embedding_model_dims = vector_size
        if distance:
            self.config.distance = _normalize_distance(distance)

    def insert(self, vectors: list[Any], payloads: list[dict[str, Any]] | None = None, ids: list[str] | None = None) -> list[str]:
        payload_list = [_normalize_memory_payload(payload) for payload in (payloads or [{} for _ in vectors])]
        return self.store.add_many(vectors, payload_list, ids)

    def search(
        self,
        query: str,
        vectors: list[float] | list[list[float]],
        top_k: int = 5,
        **_: Any,
    ) -> list[SearchHit]:
        requested_top_k = max(1, int(top_k or 5))
        query_vector = _first_vector(vectors)
        hits, metrics = self.store.search(query_vector, top_k=requested_top_k)
        self.last_search_metrics = metrics
        return [
            SearchHit(id=hit.id, payload=hit.payload, score=hit.score, distance=hit.distance, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]

    def exact_search(
        self,
        query: str,
        vectors: list[float] | list[list[float]],
        top_k: int = 5,
        **_: Any,
    ) -> list[SearchHit]:
        exact_search = getattr(self.store, "exact_search", None)
        if not callable(exact_search):
            raise RuntimeError(f"{type(self.store).__name__} does not support exact_search diagnostics.")
        query_vector = _first_vector(vectors)
        hits, _metrics = exact_search(query_vector, top_k=max(1, int(top_k or 5)))
        return [
            SearchHit(id=hit.id, payload=hit.payload, score=hit.score, distance=hit.distance, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]

    def delete(self, vector_id: str) -> None:
        raise NotImplementedError("Mem0JasperVectorStore.delete is not used by this benchmark.")

    def update(
        self,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("Mem0JasperVectorStore.update is not used by this benchmark.")

    def get(self, vector_id: str) -> SearchHit | None:
        raise NotImplementedError("Mem0JasperVectorStore.get is not used by this benchmark.")

    def list_cols(self) -> list[str]:
        return [self.collection_name]

    def delete_col(self) -> None:
        raise NotImplementedError("Mem0JasperVectorStore.delete_col is not used by this benchmark.")

    def col_info(self) -> dict[str, Any]:
        return {
            "name": self.collection_name,
            "backend": self.config.backend,
            "vectors": self.store.vector_count,
            "embedding_dim": self.store.dim,
            "path": str(self.root),
        }

    def list(
        self,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> list[SearchHit]:
        raise NotImplementedError("Mem0JasperVectorStore.list is not used by this benchmark.")

    def reset(self) -> None:
        raise NotImplementedError("Mem0JasperVectorStore.reset is not used by this benchmark.")

    def finalize(self) -> None:
        self.store.finalize()

    def close(self) -> None:
        self.store.close()

    def _create_store(self) -> Any:
        if self.config.backend == "qdrant":
            return QdrantVectorStore(self.root, self.config)
        return JasperVectorStore(self.root, self.config)


def _first_vector(vectors: list[float] | list[list[float]]) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[0] > 0:
        return array[0]
    raise ValueError("vectors must be a one-dimensional vector or a non-empty list of vectors")


def _normalize_distance(distance: str) -> str:
    lowered = str(distance).lower()
    if lowered in {"cosine", "ip", "dot"}:
        return "ip"
    if lowered in {"euclidean", "l2"}:
        return "l2"
    return lowered


def _normalize_memory_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(payload or {})
    memory = normalized.get("memory") or normalized.get("data") or normalized.get("text") or ""
    normalized.setdefault("memory", memory)
    normalized.setdefault("data", memory)

    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
    else:
        metadata = {}

    for key in _MIRRORED_METADATA_KEYS:
        top_value = normalized.get(key)
        metadata_value = metadata.get(key)
        if top_value is None and metadata_value is not None:
            normalized[key] = metadata_value
        elif metadata_value is None and top_value is not None:
            metadata[key] = top_value

    normalized["metadata"] = metadata
    return normalized
