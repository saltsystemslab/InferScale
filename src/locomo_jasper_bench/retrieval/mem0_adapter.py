from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..runtime_paths import default_mem0_dir
from ..vector_types import SearchHit, SearchMetrics, VECTOR_DISTANCE, VectorStoreConfig
from .jasper_vector_store import JasperVectorStore
from .qdrant_vector_store import QdrantVectorStore

_MIRRORED_METADATA_KEYS = (
    "fact_id",
    "user_id",
    "sample_id",
    "created_at",
    "timestamp",
    "timestamp_epoch",
    "source_session_index",
    "source_session_id",
    "source_turn_index",
    "source_turn_id",
    "turn_id",
    "session_id",
    "turn_index",
    "speaker",
    "source_role",
    "role",
)

try:
    from mem0.vector_stores.base import VectorStoreBase
except Exception:  # pragma: no cover - mem0 is optional for local unit tests

    class VectorStoreBase:  # type: ignore[no-redef]
        pass


class Mem0JasperVectorStore(VectorStoreBase):
    """Mem0 adapter dispatching to the requested local vector backend."""

    def __init__(
        self,
        *,
        collection_name: str = "memories",
        embedding_model_dims: int | None = 1536,
        path: str | Path | None = None,
        backend: str = "jasper",
        distance: str = VECTOR_DISTANCE,
        n_neighbors: int = 64,
        alpha: float = 1.0,
        workspace_budget: str = "10GB",
        beam_width: int = 64,
    ) -> None:
        _require_inner_product(distance)
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        if path is None:
            self.root = default_mem0_dir() / collection_name
        else:
            self.root = Path(path) / collection_name
        self.config = VectorStoreConfig(
            backend=backend,
            n_neighbors=n_neighbors,
            alpha=alpha,
            workspace_budget=workspace_budget,
            beam_width=beam_width,
        )
        self.store = self._create_store()
        self.last_search_metrics = SearchMetrics(
            search_time_ms=0.0,
            vector_backend=self.config.backend,
            jasper_effective_beam_width=(self.config.beam_width if self.config.backend == "jasper" else None),
        )

    def create_col(self, name: str | None = None, vector_size: int | None = None, distance: str | None = None) -> None:
        if distance is not None:
            _require_inner_product(distance)
        if name and name != self.collection_name:
            close = getattr(self.store, "close", None)
            if callable(close):
                close()
            self.collection_name = name
            self.root = self.root.parent / name
            self.store = self._create_store()
        if vector_size is not None:
            self.embedding_model_dims = vector_size

    def insert(self, vectors: list[Any], payloads: list[dict[str, Any]] | None = None, ids: list[str] | None = None) -> list[str]:
        payload_list = [_normalize_memory_payload(payload) for payload in (payloads or [{} for _ in vectors])]
        requested_ids = list(ids) if ids is not None else [None] * len(payload_list)
        stable_ids = [
            str(payload.get("fact_id") or requested_id)
            if payload.get("fact_id") is not None or requested_id is not None
            else None
            for payload, requested_id in zip(payload_list, requested_ids)
        ]
        return self.store.add_many(
            vectors,
            payload_list,
            None if any(item_id is None for item_id in stable_ids) else stable_ids,
        )

    def search(
        self,
        query: str,
        vectors: np.ndarray | list[float] | list[list[float]],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[SearchHit]:
        requested_top_k = 5 if top_k is None else int(top_k)
        if requested_top_k < 1:
            raise ValueError("top_k must be >= 1.")
        query_vector = _first_vector(vectors)
        hits, metrics = self.store.search(query_vector, top_k=requested_top_k, filters=filters)
        count = getattr(self.store, "count", None)
        matching_count = int(count(filters)) if callable(count) else len(self.store.rows(filters))
        expected_count = min(requested_top_k, matching_count)
        _validate_search_hits(hits, expected_count=expected_count, backend=self.config.backend)
        self.last_search_metrics = metrics
        return hits

    def delete(self, vector_id: str) -> None:
        self.store.delete(str(vector_id))

    def update(
        self,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.store.update(
            str(vector_id),
            vector=vector,
            payload=_normalize_memory_payload(payload) if payload is not None else None,
        )

    def get(self, vector_id: str) -> SearchHit | None:
        return self.store.get(str(vector_id))

    def list_cols(self) -> list[str]:
        return [self.collection_name]

    def delete_col(self) -> None:
        self.store.reset()

    def col_info(self) -> dict[str, Any]:
        return {
            "name": self.collection_name,
            "backend": self.config.backend,
            "vectors": self.store.vector_count,
            "embedding_dim": self.store.dim,
            "path": str(self.root),
        }

    def memory_stats(self) -> dict[str, Any]:
        stats = getattr(self.store, "memory_stats", None)
        if callable(stats):
            return dict(stats())
        return {}

    def list(
        self,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> list[SearchHit]:
        requested = top_k if top_k is not None else limit
        rows = self.store.rows(filters)
        if requested is not None:
            rows = rows[: max(0, int(requested))]
        return [
            SearchHit(
                id=item_id,
                payload=dict(payload),
                score=0.0,
                distance=0.0,
                rank=rank,
            )
            for rank, (item_id, payload) in enumerate(rows, start=1)
        ]

    def reset(self) -> None:
        self.store.reset()

    def finalize(self) -> None:
        self.store.finalize()

    def close(self) -> None:
        self.store.close()

    def _create_store(self) -> Any:
        if self.config.backend == "jasper":
            return JasperVectorStore(self.root, self.config)
        if self.config.backend == "qdrant":
            return QdrantVectorStore(self.root, self.config)
        raise ValueError(f"Unsupported vector backend: {self.config.backend!r}.")


def _first_vector(vectors: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    if isinstance(vectors, np.ndarray):
        array = vectors if vectors.dtype == np.float32 else vectors.astype(np.float32, copy=False)
    else:
        array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[0] > 0:
        return array[0]
    raise ValueError("vectors must be a one-dimensional vector or a non-empty list of vectors")


def _require_inner_product(distance: str) -> None:
    if distance != VECTOR_DISTANCE:
        raise ValueError(
            f"Mem0 Jasper provider requires distance={VECTOR_DISTANCE!r}, got {distance!r}."
        )


def _validate_search_hits(hits: list[SearchHit], *, expected_count: int, backend: str) -> None:
    if len(hits) != expected_count:
        raise RuntimeError(
            f"{backend} returned {len(hits)} hits, expected {expected_count}; refusing to use incomplete retrieval."
        )

    ids = [hit.id for hit in hits]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{backend} returned duplicate result ids; refusing to use corrupt retrieval.")

    ranks = [hit.rank for hit in hits]
    expected_ranks = list(range(1, expected_count + 1))
    if ranks != expected_ranks:
        raise RuntimeError(
            f"{backend} returned non-contiguous ranks {ranks[:10]!r}; expected ranks starting at 1."
        )

    for hit in hits:
        if not np.isfinite(hit.distance) or not np.isfinite(hit.score):
            raise RuntimeError(f"{backend} returned a non-finite score or distance for result {hit.id!r}.")
        if abs(hit.distance) >= 1e30 or abs(hit.score) >= 1e30:
            raise RuntimeError(f"{backend} returned a sentinel-like score or distance for result {hit.id!r}.")


def _normalize_memory_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(payload or {})
    memory = normalized.get("memory") or normalized.get("data") or normalized.get("text") or ""
    normalized.setdefault("memory", memory)
    normalized.setdefault("data", memory)
    normalized.setdefault("text", memory)

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
