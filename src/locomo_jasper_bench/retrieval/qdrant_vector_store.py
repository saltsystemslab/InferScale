from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..vector_types import SearchHit, SearchMetrics, VectorStoreConfig
from .store_utils import payload_matches


class QdrantVectorStore:
    """Local qdrant-client backend with the same interface as JasperVectorStore."""

    _ID_PAYLOAD_KEY = "__locomo_bench_id"
    _UUID_NAMESPACE = uuid.UUID("5c6ab2ac-d6ef-4e8f-97a7-85d8c528a0b1")

    def __init__(self, root: str | Path, config: VectorStoreConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self._collection_name = "memories"
        self._client = self._create_client()
        self._dim = self._load_dim()
        self._rows_cache: list[tuple[str, dict[str, Any]]] | None = None

    @property
    def vector_count(self) -> int:
        try:
            result = self._client.count(collection_name=self._collection_name, exact=True)
        except Exception:
            return 0
        return int(getattr(result, "count", 0) or 0)

    @property
    def dim(self) -> int | None:
        return self._dim

    def count(self, filters: dict[str, Any] | None = None) -> int:
        if self._rows_cache is not None:
            if not filters:
                return len(self._rows_cache)
            return sum(
                1
                for _, payload in self._rows_cache
                if payload_matches(payload, filters)
            )
        if self.vector_count == 0:
            return 0
        if not filters:
            return self.vector_count
        query_filter = self._native_filter(filters)
        if query_filter is None:
            return len(self.rows(filters))
        result = self._client.count(
            collection_name=self._collection_name,
            count_filter=query_filter,
            exact=True,
        )
        return int(getattr(result, "count", 0) or 0)

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

        models = self._models()
        points = []
        for item_id, vector, payload in zip(id_list, matrix, payload_list):
            next_payload = dict(payload)
            next_payload[self._ID_PAYLOAD_KEY] = str(item_id)
            points.append(
                models.PointStruct(
                    id=self._point_id(str(item_id)),
                    vector=vector.tolist(),
                    payload=next_payload,
                )
            )
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
        candidate_k = vector_count if filters and query_filter is None else top_k

        started = time.perf_counter()
        query_kwargs: dict[str, Any] = {
            "collection_name": self._collection_name,
            "query": query.tolist(),
            "limit": candidate_k,
            "with_payload": True,
        }
        if query_filter is not None:
            query_kwargs["query_filter"] = query_filter
        result = self._client.query_points(
            **query_kwargs,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        points = getattr(result, "points", result)
        hits = [
            self._hit_from_point(point, rank)
            for rank, point in enumerate(points, start=1)
            if payload_matches(self._payload_from_qdrant(getattr(point, "payload", None)), filters)
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
            next_payload = dict(payload)
            next_payload[self._ID_PAYLOAD_KEY] = str(item_id)
            self._client.set_payload(
                collection_name=self._collection_name,
                payload=next_payload,
                points=[point_id],
            )
            self._rows_cache = None
        if vector is not None:
            models = self._models()
            self._client.update_vectors(
                collection_name=self._collection_name,
                points=[
                    models.PointVectors(
                        id=point_id,
                        vector=np.asarray(vector, dtype=np.float32).tolist(),
                    )
                ],
            )

    def delete(self, item_id: str) -> None:
        models = self._models()
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[self._point_id(str(item_id))]),
        )
        self._rows_cache = None

    def reset(self) -> None:
        if self._collection_exists():
            self._client.delete_collection(collection_name=self._collection_name)
        self._dim = None
        self._rows_cache = None

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _create_client(self) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to use --vector-backend qdrant.") from exc
        return QdrantClient(path=str(self.root / "qdrant"))

    def _models(self) -> Any:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to use --vector-backend qdrant.") from exc
        return models

    def _load_dim(self) -> int | None:
        try:
            info = self._client.get_collection(collection_name=self._collection_name)
        except Exception:
            return None
        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        return getattr(vectors, "size", None)

    def _ensure_collection(self, dim: int) -> None:
        if self._dim is not None:
            if self._dim != dim:
                raise ValueError(f"vector dim {dim} does not match store dim {self._dim}")
            return

        models = self._models()
        if self._collection_exists():
            self._client.delete_collection(collection_name=self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT),
        )
        self._dim = dim

    def _collection_exists(self) -> bool:
        exists = getattr(self._client, "collection_exists", None)
        if callable(exists):
            return bool(exists(collection_name=self._collection_name))
        try:
            self._client.get_collection(collection_name=self._collection_name)
        except Exception:
            return False
        return True

    def _native_filter(self, filters: dict[str, Any] | None) -> Any | None:
        if not filters:
            return None
        models = self._models()
        must: list[Any] = []
        must_not: list[Any] = []
        for key, expected in filters.items():
            if isinstance(expected, dict):
                if not expected or not set(expected).issubset({"eq", "in", "ne"}):
                    return None
                if "eq" in expected:
                    match = _qdrant_match(models, expected["eq"])
                    if match is None:
                        return None
                    must.append(models.FieldCondition(key=key, match=match))
                if "in" in expected:
                    values = expected["in"]
                    if not isinstance(values, list) or not values or not all(
                        isinstance(value, (str, int)) and not isinstance(value, bool)
                        for value in values
                    ):
                        return None
                    must.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=values),
                        )
                    )
                if "ne" in expected:
                    match = _qdrant_match(models, expected["ne"])
                    if match is None:
                        return None
                    must_not.append(models.FieldCondition(key=key, match=match))
                continue
            match = _qdrant_match(models, expected)
            if match is None:
                return None
            must.append(models.FieldCondition(key=key, match=match))
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
        self._rows_cache = []
        for point in points:
            raw_payload = getattr(point, "payload", None)
            payload = self._payload_from_qdrant(raw_payload)
            self._rows_cache.append(
                (
                    self._original_id(getattr(point, "id", None), raw_payload),
                    payload,
                )
            )
        return self._rows_cache

    def _hit_from_point(self, point: Any, rank: int) -> SearchHit:
        raw_payload = getattr(point, "payload", None)
        raw_score = float(getattr(point, "score", 0.0) or 0.0)
        item_id = self._original_id(getattr(point, "id", None), raw_payload)
        payload = self._payload_from_qdrant(raw_payload)
        return SearchHit(
            id=item_id,
            payload=payload,
            score=raw_score,
            distance=-raw_score,
            rank=rank,
        )

    def _payload_from_qdrant(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        cleaned = dict(payload)
        cleaned.pop(self._ID_PAYLOAD_KEY, None)
        return cleaned

    def _original_id(self, point_id: Any, payload: Any) -> str:
        if isinstance(payload, dict):
            original = payload.get(self._ID_PAYLOAD_KEY)
            if original is not None:
                return str(original)
        return str(point_id)

    def _point_id(self, item_id: str) -> str:
        return str(uuid.uuid5(self._UUID_NAMESPACE, item_id))


def _qdrant_match(models: Any, value: Any) -> Any | None:
    if not isinstance(value, (str, int, bool)):
        return None
    return models.MatchValue(value=value)
