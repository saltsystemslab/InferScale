from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mem0.vector_stores.base import VectorStoreBase

from benchmarks.common.vector_types import (
    VECTOR_DISTANCE,
    SearchHit,
    SearchMetrics,
    VectorStoreConfig,
)
from benchmarks.memory.mem0.fact_catalog import FACT_METADATA_KEYS
from benchmarks.memory.mem0.search_contract import (
    first_vector,
    require_inner_product,
    validate_search_hits,
)
from benchmarks.memory.qdrant_index import QdrantVectorStore
from inferscale.v1.index.jasper import JasperIndex
from inferscale.v1.types import DeviceSearchResult

DEFAULT_TOP_K = 5


class Mem0JasperVectorStore(VectorStoreBase):
    def __init__(
        self,
        *,
        path: str | Path,
        collection_name: str = "memories",
        embedding_model_dims: int | None = 1536,
        backend: str = "jasper",
        distance: str = VECTOR_DISTANCE,
        n_neighbors: int = 64,
        alpha: float = 1.0,
        workspace_budget: str = "10GB",
        beam_width: int = 64,
    ) -> None:
        require_inner_product(distance)
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
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
            jasper_effective_beam_width=self.config.beam_width if self.is_jasper else None,
        )

    @property
    def is_jasper(self) -> bool:
        return self.config.backend == "jasper"

    def create_col(
        self, name: str | None = None, vector_size: int | None = None, distance: str | None = None
    ) -> None:
        if distance is not None:
            require_inner_product(distance)
        if name and name != self.collection_name:
            self.store.close()
            self.collection_name = name
            self.root = self.root.parent / name
            self.store = self._create_store()
        if vector_size is not None:
            self.embedding_model_dims = vector_size

    def insert(
        self,
        vectors: list[Any],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        normalized_payloads = [
            _normalize_memory_payload(payload) for payload in (payloads or [{} for _ in vectors])
        ]
        stable_ids = _stable_ids(normalized_payloads, ids)
        return self.store.add_many(vectors, normalized_payloads, stable_ids)

    def search(
        self,
        query: str,
        vectors: np.ndarray | list[float] | list[list[float]],
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[SearchHit]:
        del query
        requested = _requested_top_k(top_k)
        hits, metrics = self.store.search(first_vector(vectors), top_k=requested, filters=filters)
        expected_count = min(requested, self.store.count(filters))
        validate_search_hits(hits, expected_count=expected_count, backend=self.config.backend)
        self.last_search_metrics = metrics
        return hits

    def search_device(
        self,
        query: str,
        vectors: np.ndarray | list[float] | list[list[float]],
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
        **_: Any,
    ) -> DeviceSearchResult | None:
        del query
        requested = _requested_top_k(top_k)
        if not self.is_jasper:
            return None
        result = self.store.search_device(first_vector(vectors), top_k=requested, filters=filters)
        if result is not None:
            self.last_search_metrics = result.metrics
        return result

    def materialize_device_result(self, result: DeviceSearchResult) -> list[SearchHit]:
        if not self.is_jasper:
            raise RuntimeError("Only the Jasper backend materializes device results.")
        hits = list(self.store.materialize_device_result(result))
        expected_count = int(result.stable_ids.numel())
        validate_search_hits(hits, expected_count=expected_count, backend=self.config.backend)
        return hits

    def stable_id_items(self) -> tuple[tuple[int, str], ...]:
        if not self.is_jasper:
            raise RuntimeError("Only the Jasper backend exposes stable ids.")
        return tuple(self.store.stable_id_items())

    def memory_stats(self) -> dict[str, Any]:
        if not self.is_jasper:
            return {}
        return dict(self.store.memory_stats())

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

    def list(
        self,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> list[SearchHit]:
        row_limit = top_k if top_k is not None else limit
        rows = self.store.rows(filters)
        if row_limit is not None:
            rows = rows[: max(0, int(row_limit))]
        return [
            SearchHit(id=item_id, payload=dict(payload), score=0.0, distance=0.0, rank=rank)
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
            return JasperIndex(self.config.jasper())
        if self.config.backend == "qdrant":
            return QdrantVectorStore(self.root, self.config)
        raise ValueError(f"Unsupported vector backend: {self.config.backend!r}.")


def _requested_top_k(top_k: int | None) -> int:
    requested = DEFAULT_TOP_K if top_k is None else int(top_k)
    if requested < 1:
        raise ValueError("top_k must be >= 1.")
    return requested


def _stable_ids(payloads: list[dict[str, Any]], ids: list[str] | None) -> list[str] | None:
    requested_ids = list(ids) if ids is not None else [None] * len(payloads)
    if len(requested_ids) != len(payloads):
        raise ValueError("ids and payloads must have the same length")
    stable_ids = [
        payload.get("fact_id") if payload.get("fact_id") is not None else requested_id
        for payload, requested_id in zip(payloads, requested_ids)
    ]
    store_must_generate_ids = any(stable_id is None for stable_id in stable_ids)
    if store_must_generate_ids:
        return None
    stable_id_texts = [str(stable_id) for stable_id in stable_ids]
    if "" in stable_id_texts:
        raise ValueError("Mem0 vector ids must not be empty.")
    return stable_id_texts


def _normalize_memory_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(payload or {})
    memory_text = normalized.get("memory") or normalized.get("data") or normalized.get("text") or ""
    normalized.setdefault("memory", memory_text)
    normalized.setdefault("data", memory_text)
    normalized.setdefault("text", memory_text)
    raw_metadata = normalized.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    _mirror_fact_metadata(normalized, metadata)
    normalized["metadata"] = metadata
    return normalized


def _mirror_fact_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    for key in FACT_METADATA_KEYS:
        top_level_value = payload.get(key)
        metadata_value = metadata.get(key)
        if top_level_value is None and metadata_value is not None:
            payload[key] = metadata_value
        elif metadata_value is None and top_level_value is not None:
            metadata[key] = top_level_value
