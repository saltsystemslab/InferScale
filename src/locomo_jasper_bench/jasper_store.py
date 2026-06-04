from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(slots=True)
class VectorStoreConfig:
    backend: str = "jasper"
    distance: str = "ip"
    normalize: bool = True
    n_neighbors: int = 64
    alpha: float = 1.2
    workspace_budget: str = "10GB"
    beam_width: int = 64


@dataclass(slots=True)
class BuildMetrics:
    backend: str
    graph_build_time_ms: float
    indexed_vector_count: int
    embedding_dim: int | None
    graph_path: str | None = None


@dataclass(slots=True)
class SearchMetrics:
    backend: str
    search_time_ms: float
    indexed_vector_count: int
    embedding_dim: int | None


@dataclass(slots=True)
class SearchHit:
    id: str
    payload: dict[str, Any]
    score: float
    distance: float
    rank: int


class JasperVectorStore:
    """File-backed vector store with Jasper GPU search and NumPy exact-search fallback."""

    def __init__(self, root: str | Path, config: VectorStoreConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self.root / "vectors.npy"
        self._graph_path = self.root / "jasper.graph"
        self._db_path = self.root / "payloads.sqlite"
        self._conn = sqlite3.connect(self._db_path, timeout=30.0)
        self._init_db()
        self._vectors: np.ndarray | None = self._load_vectors()
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
        if self.config.normalize:
            matrix = _normalize_rows(matrix)
        if self._vectors is None:
            self._vectors = matrix
            start_ord = 0
        else:
            if matrix.shape[1] != self._vectors.shape[1]:
                raise ValueError(f"vector dim {matrix.shape[1]} does not match store dim {self._vectors.shape[1]}")
            start_ord = int(self._vectors.shape[0])
            self._vectors = np.vstack([self._vectors, matrix]).astype(np.float32, copy=False)

        try:
            with self._conn:
                for offset, (item_id, payload) in enumerate(zip(id_list, payload_list)):
                    self._conn.execute(
                        "INSERT OR REPLACE INTO payloads (id, ord, payload_json) VALUES (?, ?, ?)",
                        (item_id, start_ord + offset, json.dumps(payload, ensure_ascii=False)),
                    )
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"Could not write Jasper payload database at {self._db_path}. "
                "On shared clusters this is usually caused by project quota, permissions, or SQLite journal/locking "
                "support on the target filesystem. Check free space and quota for the results directory, and rerun "
                "with --results-dir and BENCHMARK_CACHE_ROOT pointing at a writable project filesystem."
            ) from exc
        self._graph = None
        return id_list

    def finalize(self) -> BuildMetrics:
        if self._vectors is None or self._vectors.size == 0:
            return BuildMetrics(
                backend=self.config.backend,
                graph_build_time_ms=0.0,
                indexed_vector_count=0,
                embedding_dim=None,
            )

        np.save(self._vectors_path, self._vectors)
        if self.config.backend == "numpy":
            return BuildMetrics(
                backend="numpy",
                graph_build_time_ms=0.0,
                indexed_vector_count=self.vector_count,
                embedding_dim=self.dim,
            )
        if self.config.backend != "jasper":
            raise ValueError(f"Unsupported vector backend: {self.config.backend}")

        started = time.perf_counter()
        self._graph = self._build_jasper_graph()
        build_time_ms = (time.perf_counter() - started) * 1000
        if self._graph_path:
            self._graph.save(str(self._graph_path))
        return BuildMetrics(
            backend="jasper",
            graph_build_time_ms=build_time_ms,
            indexed_vector_count=self.vector_count,
            embedding_dim=self.dim,
            graph_path=str(self._graph_path),
        )

    def search(self, query_vector: np.ndarray | list[float], top_k: int) -> tuple[list[SearchHit], SearchMetrics]:
        if self._vectors is None or self._vectors.size == 0:
            return [], SearchMetrics(self.config.backend, 0.0, 0, None)
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self._vectors.shape[1]}")
        if self.config.normalize:
            query = _normalize_vector(query)
        top_k = max(1, min(top_k, self.vector_count))

        started = time.perf_counter()
        if self.config.backend == "jasper":
            hits = self._search_jasper(query, top_k)
        elif self.config.backend == "numpy":
            hits = self._search_numpy(query, top_k)
        else:
            raise ValueError(f"Unsupported vector backend: {self.config.backend}")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return hits, SearchMetrics(
            backend=self.config.backend,
            search_time_ms=elapsed_ms,
            indexed_vector_count=self.vector_count,
            embedding_dim=self.dim,
        )

    def close(self) -> None:
        if self._graph is not None:
            free = getattr(self._graph, "free", None)
            if callable(free):
                free()
            self._graph = None
        self._conn.close()

    def delete(self, vector_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM payloads WHERE id = ?", (str(vector_id),))
        self._graph = None

    def update(
        self,
        vector_id: str,
        vector: np.ndarray | list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        row = self._conn.execute(
            "SELECT ord, payload_json FROM payloads WHERE id = ?",
            (str(vector_id),),
        ).fetchone()
        if row is None:
            if vector is None:
                return
            self.add_many([vector], [payload or {}], [str(vector_id)])
            return

        ordinal = int(row[0])
        current_payload = json.loads(row[1])
        next_payload = current_payload if payload is None else payload
        if vector is not None and self._vectors is not None:
            next_vector = np.asarray(vector, dtype=np.float32)
            if self.config.normalize:
                next_vector = _normalize_vector(next_vector)
            self._vectors[ordinal] = next_vector
        with self._conn:
            self._conn.execute(
                "UPDATE payloads SET payload_json = ? WHERE id = ?",
                (json.dumps(next_payload, ensure_ascii=False), str(vector_id)),
            )
        self._graph = None

    def get(self, vector_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload_json FROM payloads WHERE id = ?",
            (str(vector_id),),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def list_payloads(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        rows = self._conn.execute(
            "SELECT id, payload_json FROM payloads ORDER BY ord LIMIT ?",
            (limit,),
        ).fetchall()
        return [(str(item_id), json.loads(payload_json)) for item_id, payload_json in rows]

    def reset(self) -> None:
        self.close()
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, timeout=30.0)
        self._init_db()
        self._vectors = None
        self._graph = None

    def _init_db(self) -> None:
        self._configure_db()
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS payloads (id TEXT PRIMARY KEY, ord INTEGER UNIQUE NOT NULL, payload_json TEXT NOT NULL)"
            )

    def _configure_db(self) -> None:
        # Avoid sidecar journal/temp files; they are fragile on some shared HPC filesystems.
        self._conn.execute("PRAGMA journal_mode=MEMORY")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA busy_timeout=30000")

    def _load_vectors(self) -> np.ndarray | None:
        if self._vectors_path.exists():
            return np.load(self._vectors_path).astype(np.float32, copy=False)
        return None

    def _build_jasper_graph(self) -> Any:
        try:
            import torch
            import jasper
        except ImportError as exc:
            raise RuntimeError(
                "Jasper backend requires torch, apache-tvm-ffi, and the built jasper Python package. "
                "Use --vector-backend numpy for CPU-only dry runs."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Jasper backend requires a CUDA device. Use --vector-backend numpy for CPU-only dry runs.")
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
            if self._graph_path.exists():
                self._graph = self._load_jasper_graph()
            else:
                self._graph = self._build_jasper_graph()
        import torch

        query_tensor = torch.from_numpy(query.reshape(1, -1)).to(device="cuda", dtype=torch.float32)
        indices, distances = self._graph.search(query_tensor, k=top_k, beam_width=self.config.beam_width)
        index_values = indices[0].detach().cpu().numpy().astype(np.int64)
        distance_values = distances[0].detach().cpu().numpy().astype(np.float32)
        return self._hits_from_ordinals(index_values, distance_values)

    def _load_jasper_graph(self) -> Any:
        try:
            import jasper
        except ImportError as exc:
            raise RuntimeError("Could not import jasper to load the graph.") from exc
        return jasper.Graph.load(
            str(self._graph_path),
            dim=int(self.dim or 0),
            n_neighbors=self.config.n_neighbors,
            distance=self.config.distance,
        )

    def _search_numpy(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        assert self._vectors is not None
        if self.config.distance == "ip":
            raw_scores = self._vectors @ query
            ordinals = np.argsort(-raw_scores)[:top_k]
            distances = raw_scores[ordinals]
        elif self.config.distance == "l2":
            raw_distances = np.sum((self._vectors - query) ** 2, axis=1)
            ordinals = np.argsort(raw_distances)[:top_k]
            distances = raw_distances[ordinals]
        else:
            raise ValueError(f"Unsupported distance metric: {self.config.distance}")
        return self._hits_from_ordinals(ordinals.astype(np.int64), distances.astype(np.float32))

    def _hits_from_ordinals(self, ordinals: np.ndarray, distances: np.ndarray) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for rank, (ordinal, distance) in enumerate(zip(ordinals.tolist(), distances.tolist()), start=1):
            row = self._payload_by_ordinal(int(ordinal))
            if row is None:
                continue
            item_id, payload = row
            score = float(distance if self.config.distance == "ip" else -distance)
            hits.append(SearchHit(id=item_id, payload=payload, score=score, distance=float(distance), rank=rank))
        return hits

    def _payload_by_ordinal(self, ordinal: int) -> tuple[str, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT id, payload_json FROM payloads WHERE ord = ?",
            (ordinal,),
        ).fetchone()
        if row is None:
            return None
        return str(row[0]), json.loads(row[1])


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


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
        if self.config.normalize:
            matrix = _normalize_rows(matrix)
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

    def finalize(self) -> BuildMetrics:
        return BuildMetrics(
            backend="qdrant",
            graph_build_time_ms=0.0,
            indexed_vector_count=self.vector_count,
            embedding_dim=self.dim,
            graph_path=None,
        )

    def search(self, query_vector: np.ndarray | list[float], top_k: int) -> tuple[list[SearchHit], SearchMetrics]:
        if self.vector_count == 0:
            return [], SearchMetrics("qdrant", 0.0, 0, self.dim)
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if self.dim is not None and query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self.dim}")
        if self.config.normalize:
            query = _normalize_vector(query)
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
        return hits, SearchMetrics(
            backend="qdrant",
            search_time_ms=elapsed_ms,
            indexed_vector_count=self.vector_count,
            embedding_dim=self.dim,
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def delete(self, vector_id: str) -> None:
        models = self._models()
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[self._point_id(str(vector_id))]),
        )

    def update(
        self,
        vector_id: str,
        vector: np.ndarray | list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        existing = self.get(vector_id)
        if existing is None and vector is None:
            return
        next_payload = existing or {}
        if payload is not None:
            next_payload = payload
        if vector is not None:
            self.add_many([vector], [next_payload], [str(vector_id)])
            return
        next_payload = dict(next_payload)
        next_payload[self._ID_PAYLOAD_KEY] = str(vector_id)
        overwrite_payload = getattr(self._client, "overwrite_payload", None)
        if callable(overwrite_payload):
            overwrite_payload(
                collection_name=self._collection_name,
                payload=next_payload,
                points=[self._point_id(str(vector_id))],
            )
        else:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload=next_payload,
                points=[self._point_id(str(vector_id))],
            )

    def get(self, vector_id: str) -> dict[str, Any] | None:
        try:
            points = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[self._point_id(str(vector_id))],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return None
        if not points:
            return None
        return self._payload_from_qdrant(getattr(points[0], "payload", None))

    def list_payloads(self, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return []
        results: list[tuple[str, dict[str, Any]]] = []
        for point in points:
            raw_payload = getattr(point, "payload", None)
            item_id = self._original_id(getattr(point, "id", None), raw_payload)
            payload = self._payload_from_qdrant(raw_payload)
            results.append((item_id, payload))
        return results

    def reset(self) -> None:
        self.close()
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self._client = self._create_client()
        self._dim = None

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
        raw_score = float(getattr(point, "score", 0.0) or 0.0)
        raw_payload = getattr(point, "payload", None)
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
