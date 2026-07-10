from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..vector_types import SearchHit, SearchMetrics, VectorStoreConfig


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
        return [str(item_id) for item_id in id_list]

    def finalize(self) -> None:
        return

    def search(self, query_vector: np.ndarray | list[float], top_k: int) -> tuple[list[SearchHit], SearchMetrics]:
        if self.vector_count == 0:
            return [], SearchMetrics(0.0, vector_backend="qdrant")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if self.dim is not None and query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self.dim}")
        top_k = max(1, min(top_k, self.vector_count))

        started = time.perf_counter()
        result = self._client.query_points(
            collection_name=self._collection_name,
            query=query.tolist(),
            limit=top_k,
            with_payload=True,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        points = getattr(result, "points", result)
        hits = [self._hit_from_point(point, rank) for rank, point in enumerate(points, start=1)]
        return hits, SearchMetrics(elapsed_ms, vector_backend="qdrant")

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
        distance = models.Distance.DOT if self.config.distance == "ip" else models.Distance.EUCLID
        if self._collection_exists():
            self._client.delete_collection(collection_name=self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=dim, distance=distance),
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

    def _hit_from_point(self, point: Any, rank: int) -> SearchHit:
        raw_payload = getattr(point, "payload", None)
        raw_score = float(getattr(point, "score", 0.0) or 0.0)
        item_id = self._original_id(getattr(point, "id", None), raw_payload)
        payload = self._payload_from_qdrant(raw_payload)
        if self.config.distance == "ip":
            score = raw_score
            distance = raw_score
        else:
            score = -raw_score
            distance = raw_score
        return SearchHit(id=item_id, payload=payload, score=score, distance=distance, rank=rank)

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
