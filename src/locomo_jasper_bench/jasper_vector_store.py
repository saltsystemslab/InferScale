from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .vector_types import SearchHit, SearchMetrics, VectorStoreConfig


class JasperVectorStore:
    """File-backed vector store with Jasper GPU search."""

    def __init__(self, root: str | Path, config: VectorStoreConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self.root / "vectors.npy"
        self._graph_path = self.root / "jasper.graph"
        self._db_path = self.root / "payloads.sqlite"
        self._conn = sqlite3.connect(self._db_path, timeout=30.0)
        self._init_db()
        self._payloads_by_ordinal: dict[int, tuple[str, dict[str, Any]]] = {}
        self._ordinals_by_id: dict[str, int] = {}
        self._load_payload_cache()
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
        if self._vectors is None:
            self._vectors = matrix
            start_ord = 0
        else:
            if matrix.shape[1] != self._vectors.shape[1]:
                raise ValueError(f"vector dim {matrix.shape[1]} does not match store dim {self._vectors.shape[1]}")
            start_ord = int(self._vectors.shape[0])
            self._vectors = np.vstack([self._vectors, matrix]).astype(np.float32, copy=False)

        cache_updates: list[tuple[str, int, dict[str, Any]]] = []
        try:
            with self._conn:
                for offset, (item_id, payload) in enumerate(zip(id_list, payload_list)):
                    payload_json = json.dumps(payload, ensure_ascii=False)
                    self._conn.execute(
                        "INSERT OR REPLACE INTO payloads (id, ord, payload_json) VALUES (?, ?, ?)",
                        (item_id, start_ord + offset, payload_json),
                    )
                    cache_updates.append((str(item_id), start_ord + offset, json.loads(payload_json)))
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"Could not write Jasper payload database at {self._db_path}. "
                "On shared clusters this is usually caused by project quota, permissions, or SQLite journal/locking "
                "support on the target filesystem. Check free space and quota for the results directory, and rerun "
                "with --results-dir and BENCHMARK_CACHE_ROOT pointing at a writable project filesystem."
            ) from exc
        for item_id, ordinal, payload in cache_updates:
            old_ordinal = self._ordinals_by_id.get(item_id)
            if old_ordinal is not None:
                self._payloads_by_ordinal.pop(old_ordinal, None)
            self._ordinals_by_id[item_id] = ordinal
            self._payloads_by_ordinal[ordinal] = (item_id, payload)
        self._graph = None
        return id_list

    def finalize(self) -> None:
        if self._vectors is None or self._vectors.size == 0:
            return

        if self.config.backend != "jasper":
            raise ValueError(f"Unsupported vector backend: {self.config.backend}")

        np.save(self._vectors_path, self._vectors)
        self._graph = self._build_jasper_graph()
        if self._graph_path:
            self._graph.save(str(self._graph_path))

    def search(self, query_vector: np.ndarray | list[float], top_k: int) -> tuple[list[SearchHit], SearchMetrics]:
        if self._vectors is None or self._vectors.size == 0:
            return [], SearchMetrics(0.0)
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self._vectors.shape[1]}")
        top_k = max(1, min(top_k, self.vector_count))

        if self.config.backend == "jasper":
            hits, search_time_ms = self._search_jasper(query, top_k)
        else:
            raise ValueError(f"Unsupported vector backend: {self.config.backend}")
        return hits, SearchMetrics(search_time_ms)

    def close(self) -> None:
        if self._graph is not None:
            free = getattr(self._graph, "free", None)
            if callable(free):
                free()
            self._graph = None
        self._conn.close()

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

    def _load_payload_cache(self) -> None:
        rows = self._conn.execute("SELECT id, ord, payload_json FROM payloads").fetchall()
        self._payloads_by_ordinal = {}
        self._ordinals_by_id = {}
        for item_id, ordinal, payload_json in rows:
            item_id = str(item_id)
            ordinal = int(ordinal)
            payload = json.loads(payload_json)
            self._payloads_by_ordinal[ordinal] = (item_id, payload)
            self._ordinals_by_id[item_id] = ordinal

    def _build_jasper_graph(self) -> Any:
        try:
            import torch
            import jasper
        except ImportError as exc:
            raise RuntimeError(
                "Jasper backend requires torch, apache-tvm-ffi, and the built jasper Python package. "
                "Use --vector-backend qdrant for CPU-only vector-search comparisons."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Jasper backend requires a CUDA device. Use --vector-backend qdrant for CPU-only vector-search comparisons."
            )
        vectors = torch.from_numpy(self._vectors).to(device="cuda", dtype=torch.float32)

        return jasper.Graph.build(
            vectors,
            n_neighbors=self.config.n_neighbors,
            distance=self.config.distance,
            alpha=self.config.alpha,
            workspace_budget=self.config.workspace_budget,
        )

    def _search_jasper(self, query: np.ndarray, top_k: int) -> tuple[list[SearchHit], float]:
        if self._graph is None:
            if self._graph_path.exists():
                self._graph = self._load_jasper_graph()
            else:
                self._graph = self._build_jasper_graph()
        import torch

        query_tensor = torch.from_numpy(query.reshape(1, -1)).to(device="cuda", dtype=torch.float32)
        synchronize = getattr(getattr(torch, "cuda", None), "synchronize", None)
        if callable(synchronize):
            synchronize()
        started = time.perf_counter()
        indices, distances = self._graph.search(query_tensor, k=top_k, beam_width=self.config.beam_width)
        if callable(synchronize):
            synchronize()
        search_time_ms = (time.perf_counter() - started) * 1000

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
        return hits, search_time_ms

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

    def _payload_by_ordinal(self, ordinal: int) -> tuple[str, dict[str, Any]] | None:
        return self._payloads_by_ordinal.get(ordinal)
