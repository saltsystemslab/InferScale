"""A local qdrant-client vector store with the same interface as JasperIndex."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common.vector_types import SearchHit, SearchMetrics, VectorStoreConfig
from inferscale.v1.index.filters import payload_matches

FILTER_OPERATORS = {"eq", "in", "ne"}


class QdrantVectorStore:
    """Vectors and payloads in an on-disk Qdrant collection, searched by inner product."""

    _ID_PAYLOAD_KEY = "__locomo_bench_id"
    _UUID_NAMESPACE = uuid.UUID("5c6ab2ac-d6ef-4e8f-97a7-85d8c528a0b1")

    def __init__(self, root: str | Path, config: VectorStoreConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self._collection_name = "memories"
        self._client = _qdrant_client(self.root / "qdrant")
        self._dim = self._load_dim()
        # Filled by finalize(); serves rows() and count() without touching the client.
        self._rows_cache: list[tuple[str, dict[str, Any]]] | None = None

    @property
    def vector_count(self) -> int:
        try:
            result = self._client.count(collection_name=self._collection_name, exact=True)
        except Exception:
            return 0
        return int(result.count or 0)

    @property
    def dim(self) -> int | None:
        return self._dim

    def count(self, filters: dict[str, Any] | None = None) -> int:
        if self._rows_cache is not None:
            return sum(1 for _, payload in self._rows_cache if payload_matches(payload, filters))
        if self.vector_count == 0:
            return 0
        if not filters:
            return self.vector_count
        query_filter = self._native_filter(filters)
        if query_filter is None:
            return len(self.rows(filters))
        result = self._client.count(
            collection_name=self._collection_name, count_filter=query_filter, exact=True
        )
        return int(result.count or 0)

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
        self._ensure_collection(int(matrix.shape[1]))
        models = _qdrant_models()
        points = [
            models.PointStruct(
                id=self._point_id(str(item_id)),
                vector=vector.tolist(),
                payload={**payload, self._ID_PAYLOAD_KEY: str(item_id)},
            )
            for item_id, vector, payload in zip(id_list, matrix, payload_list)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)
        self._rows_cache = None
        return [str(item_id) for item_id in id_list]

    def finalize(self) -> None:
        self._load_rows()

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchHit], SearchMetrics]:
        vector_count = self.vector_count
        if vector_count == 0:
            return [], SearchMetrics(0.0, vector_backend="qdrant")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if self.dim is not None and query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self.dim}")
        top_k = max(1, min(top_k, vector_count))
        query_filter = self._native_filter(filters)
        # A filter Qdrant cannot evaluate is applied here, after fetching every vector.
        candidate_k = vector_count if filters and query_filter is None else top_k
        query_kwargs: dict[str, Any] = {
            "collection_name": self._collection_name,
            "query": query.tolist(),
            "limit": candidate_k,
            "with_payload": True,
        }
        if query_filter is not None:
            query_kwargs["query_filter"] = query_filter
        started = time.perf_counter()
        result = self._client.query_points(**query_kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000
        hits = [
            self._hit_from_point(point, rank)
            for rank, point in enumerate(result.points, start=1)
            if payload_matches(self._payload_from_qdrant(point.payload), filters)
        ][:top_k]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits, SearchMetrics(elapsed_ms, vector_backend="qdrant")

    def rows(self, filters: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
        return [
            (item_id, dict(payload))
            for item_id, payload in self._load_rows()
            if payload_matches(payload, filters)
        ]

    def get(self, item_id: str) -> SearchHit | None:
        points = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[self._point_id(str(item_id))],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        return self._hit_from_point(points[0], 1)

    def update(
        self,
        item_id: str,
        *,
        vector: np.ndarray | list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        point_id = self._point_id(str(item_id))
        if payload is not None:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={**payload, self._ID_PAYLOAD_KEY: str(item_id)},
                points=[point_id],
            )
            self._rows_cache = None
        if vector is not None:
            models = _qdrant_models()
            self._client.update_vectors(
                collection_name=self._collection_name,
                points=[
                    models.PointVectors(
                        id=point_id, vector=np.asarray(vector, dtype=np.float32).tolist()
                    )
                ],
            )

    def delete(self, item_id: str) -> None:
        models = _qdrant_models()
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[self._point_id(str(item_id))]),
        )
        self._rows_cache = None

    def reset(self) -> None:
        if self._client.collection_exists(collection_name=self._collection_name):
            self._client.delete_collection(collection_name=self._collection_name)
        self._dim = None
        self._rows_cache = None

    def close(self) -> None:
        self._client.close()

    def _load_dim(self) -> int | None:
        try:
            info = self._client.get_collection(collection_name=self._collection_name)
        except Exception:
            return None
        return info.config.params.vectors.size

    def _ensure_collection(self, dim: int) -> None:
        if self._dim is not None:
            if self._dim != dim:
                raise ValueError(f"vector dim {dim} does not match store dim {self._dim}")
            return
        models = _qdrant_models()
        if self._client.collection_exists(collection_name=self._collection_name):
            self._client.delete_collection(collection_name=self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT),
        )
        self._dim = dim

    def _native_filter(self, filters: dict[str, Any] | None) -> Any | None:
        """The filters as a Qdrant filter, or None when a clause has no Qdrant equivalent."""
        if not filters:
            return None
        models = _qdrant_models()
        must: list[Any] = []
        must_not: list[Any] = []
        for key, expected in filters.items():
            conditions = _field_conditions(models, key, expected)
            if conditions is None:
                return None
            must.extend(conditions[0])
            must_not.extend(conditions[1])
        return models.Filter(must=must or None, must_not=must_not or None)

    def _load_rows(self) -> list[tuple[str, dict[str, Any]]]:
        if self._rows_cache is not None:
            return self._rows_cache
        if self.vector_count == 0:
            self._rows_cache = []
            return self._rows_cache
        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            limit=max(1, self.vector_count),
            with_payload=True,
            with_vectors=False,
        )
        self._rows_cache = [
            (self._original_id(point.id, point.payload), self._payload_from_qdrant(point.payload))
            for point in points
        ]
        return self._rows_cache

    def _hit_from_point(self, point: Any, rank: int) -> SearchHit:
        score = float(getattr(point, "score", 0.0) or 0.0)
        return SearchHit(
            id=self._original_id(point.id, point.payload),
            payload=self._payload_from_qdrant(point.payload),
            score=score,
            distance=-score,
            rank=rank,
        )

    def _payload_from_qdrant(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        cleaned = dict(payload)
        cleaned.pop(self._ID_PAYLOAD_KEY, None)
        return cleaned

    def _original_id(self, point_id: Any, payload: Any) -> str:
        if isinstance(payload, dict) and payload.get(self._ID_PAYLOAD_KEY) is not None:
            return str(payload[self._ID_PAYLOAD_KEY])
        return str(point_id)

    def _point_id(self, item_id: str) -> str:
        return str(uuid.uuid5(self._UUID_NAMESPACE, item_id))


def _qdrant_client(path: Path) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("vector_backend=qdrant requires the qdrant-client package.") from exc
    return QdrantClient(path=str(path))


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError("vector_backend=qdrant requires the qdrant-client package.") from exc
    return models


def _field_conditions(models: Any, key: str, expected: Any) -> tuple[list[Any], list[Any]] | None:
    """Qdrant conditions (must, must_not) for one filter clause, or None if untranslatable."""
    if not isinstance(expected, dict):
        match = _match_value(models, expected)
        return ([models.FieldCondition(key=key, match=match)], []) if match is not None else None
    if not expected or not set(expected).issubset(FILTER_OPERATORS):
        return None
    must: list[Any] = []
    must_not: list[Any] = []
    if "eq" in expected:
        match = _match_value(models, expected["eq"])
        if match is None:
            return None
        must.append(models.FieldCondition(key=key, match=match))
    if "in" in expected:
        values = expected["in"]
        if not isinstance(values, list) or not values or not all(_is_scalar(v) for v in values):
            return None
        must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
    if "ne" in expected:
        match = _match_value(models, expected["ne"])
        if match is None:
            return None
        must_not.append(models.FieldCondition(key=key, match=match))
    return must, must_not


def _match_value(models: Any, value: Any) -> Any | None:
    if not isinstance(value, str | int | bool):
        return None
    return models.MatchValue(value=value)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, str | int) and not isinstance(value, bool)
