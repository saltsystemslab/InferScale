from __future__ import annotations

import json
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .jasper_store import BuildMetrics, JasperVectorStore, SearchHit, SearchMetrics, VectorStoreConfig

try:
    from mem0.vector_stores.base import VectorStoreBase
except Exception:  # pragma: no cover - mem0 is optional for local unit tests

    class VectorStoreBase:  # type: ignore[no-redef]
        pass


@dataclass(slots=True)
class Mem0JasperSearchResult:
    id: str
    score: float
    payload: dict[str, Any]
    vector: list[float] | None = None


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
        normalize: bool = True,
        n_neighbors: int = 64,
        alpha: float = 1.2,
        workspace_budget: str = "10GB",
        beam_width: int = 64,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.root = Path(path) / collection_name
        self.config = VectorStoreConfig(
            backend=backend,
            distance=distance,
            normalize=normalize,
            n_neighbors=n_neighbors,
            alpha=alpha,
            workspace_budget=workspace_budget,
            beam_width=beam_width,
        )
        self.store = JasperVectorStore(self.root, self.config)
        self.last_insert_ids: list[str] = []
        self.last_build_metrics = BuildMetrics(
            backend=backend,
            graph_build_time_ms=0.0,
            indexed_vector_count=self.store.vector_count,
            embedding_dim=self.store.dim,
            graph_path=None,
        )
        self.last_search_metrics = SearchMetrics(
            backend=backend,
            search_time_ms=0.0,
            indexed_vector_count=self.store.vector_count,
            embedding_dim=self.store.dim,
        )

    def create_col(self, name: str | None = None, vector_size: int | None = None, distance: str | None = None) -> None:
        if name:
            self.collection_name = name
        if vector_size is not None:
            self.embedding_model_dims = vector_size
        if distance:
            self.config.distance = _normalize_distance(distance)

    def insert(self, vectors: list[Any], payloads: list[dict[str, Any]] | None = None, ids: list[str] | None = None) -> list[str]:
        payload_list = payloads or [{} for _ in vectors]
        self.last_insert_ids = self.store.add_many(vectors, payload_list, ids)
        return self.last_insert_ids

    def search(
        self,
        query: str,
        vectors: list[float] | list[list[float]],
        top_k: int = 5,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[Mem0JasperSearchResult]:
        top_k = limit or top_k
        query_vector = _first_vector(vectors)
        limit = self.store.vector_count if filters else top_k
        hits, metrics = self.store.search(query_vector, top_k=max(1, limit))
        self.last_search_metrics = metrics
        if filters:
            hits = [hit for hit in hits if _matches_filters(hit.payload, filters)]
        return [
            Mem0JasperSearchResult(id=hit.id, score=hit.score, payload=hit.payload)
            for hit in _rerank_hits(hits[:top_k])
        ]

    def delete(self, vector_id: str) -> None:
        with self.store._conn:
            self.store._conn.execute("DELETE FROM payloads WHERE id = ?", (str(vector_id),))
        self.store._graph = None

    def update(
        self,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        row = self.store._conn.execute(
            "SELECT ord, payload_json FROM payloads WHERE id = ?",
            (str(vector_id),),
        ).fetchone()
        if row is None:
            if vector is None:
                return
            self.insert([vector], [payload or {}], [str(vector_id)])
            return

        ordinal = int(row[0])
        current_payload = json.loads(row[1])
        next_payload = current_payload if payload is None else payload
        if vector is not None and self.store._vectors is not None:
            next_vector = np.asarray(vector, dtype=np.float32)
            if self.config.normalize:
                next_vector = _normalize_vector(next_vector)
            self.store._vectors[ordinal] = next_vector
        with self.store._conn:
            self.store._conn.execute(
                "UPDATE payloads SET payload_json = ? WHERE id = ?",
                (json.dumps(next_payload, ensure_ascii=False), str(vector_id)),
            )
        self.store._graph = None

    def get(self, vector_id: str) -> Mem0JasperSearchResult | None:
        row = self.store._conn.execute(
            "SELECT payload_json FROM payloads WHERE id = ?",
            (str(vector_id),),
        ).fetchone()
        if row is None:
            return None
        return Mem0JasperSearchResult(id=str(vector_id), score=0.0, payload=json.loads(row[0]))

    def list_cols(self) -> list[str]:
        return [self.collection_name]

    def delete_col(self) -> None:
        self.close()
        shutil.rmtree(self.root, ignore_errors=True)
        self.store = JasperVectorStore(self.root, self.config)

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
    ) -> list[Mem0JasperSearchResult]:
        limit = limit or top_k or 100
        rows = self.store._conn.execute(
            "SELECT id, payload_json FROM payloads ORDER BY ord LIMIT ?",
            (limit,),
        ).fetchall()
        results: list[Mem0JasperSearchResult] = []
        for item_id, payload_json in rows:
            payload = json.loads(payload_json)
            if filters and not _matches_filters(payload, filters):
                continue
            results.append(Mem0JasperSearchResult(id=str(item_id), score=0.0, payload=payload))
        return results

    def reset(self) -> None:
        self.delete_col()

    def finalize(self) -> BuildMetrics:
        self.last_build_metrics = self.store.finalize()
        return self.last_build_metrics

    def close(self) -> None:
        self.store.close()


def create_mem0_memory(
    *,
    store_root: str | Path,
    vector_config: VectorStoreConfig,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
) -> Any:
    register_mem0_jasper_provider()
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError("Install the mem0ai package to use --context-mode mem0.") from exc

    store_root = Path(store_root)
    store_root.mkdir(parents=True, exist_ok=True)
    mem0_config = build_mem0_config(
        store_root=store_root,
        vector_config=vector_config,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )
    if hasattr(Memory, "from_config"):
        return Memory.from_config(mem0_config)

    from mem0.configs.base import MemoryConfig

    return Memory(MemoryConfig(**mem0_config))


def build_mem0_config(
    *,
    store_root: str | Path,
    vector_config: VectorStoreConfig,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
) -> dict[str, Any]:
    embedder_config: dict[str, Any] = {"model": embedding_model}
    if embedding_api_key:
        embedder_config["api_key"] = embedding_api_key
    if embedding_base_url:
        embedder_config["openai_base_url"] = embedding_base_url

    store_root = Path(store_root)
    return {
        "vector_store": {
            "provider": "jasper",
            "config": {
                "collection_name": "memories",
                "path": str(store_root),
                "backend": vector_config.backend,
                "distance": vector_config.distance,
                "normalize": vector_config.normalize,
                "n_neighbors": vector_config.n_neighbors,
                "alpha": vector_config.alpha,
                "workspace_budget": vector_config.workspace_budget,
                "beam_width": vector_config.beam_width,
            },
        },
        "embedder": {"provider": "openai", "config": embedder_config},
        "history_db_path": str(store_root / "history.sqlite"),
    }


def register_mem0_jasper_provider() -> None:
    try:
        from mem0.utils.factory import VectorStoreFactory
        from mem0.vector_stores.configs import VectorStoreConfig as Mem0VectorStoreConfig
    except ImportError as exc:
        raise RuntimeError("Install the mem0ai package to register the Jasper Mem0 provider.") from exc

    VectorStoreFactory.provider_to_class["jasper"] = "locomo_jasper_bench.mem0_jasper.Mem0JasperVectorStore"
    _install_jasper_config_module()
    _patch_mem0_vector_config_registry(Mem0VectorStoreConfig)


def mem0_results_to_search_hits(results: Any) -> list[SearchHit]:
    items = _mem0_result_items(results)
    hits: list[SearchHit] = []
    for rank, item in enumerate(items, start=1):
        payload = _mem0_item_payload(item)
        score = _float_value(_item_get(item, "score"), 0.0)
        hits.append(
            SearchHit(
                id=str(_item_get(item, "id") or rank),
                payload=payload,
                score=score,
                distance=_float_value(_item_get(item, "distance"), score),
                rank=rank,
            )
        )
    return hits


def _install_jasper_config_module() -> None:
    try:
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError("mem0ai requires pydantic; install mem0ai to configure the Jasper provider.") from exc

    class JasperConfig(BaseModel):
        collection_name: str = Field("memories", description="Name of the collection")
        embedding_model_dims: int | None = Field(1536, description="Dimensions of the embedding model")
        path: str = Field("/tmp/jasper", description="Path for the Jasper vector store")
        backend: str = Field("jasper", description="JasperVectorStore backend")
        distance: str = Field("ip", description="Distance metric")
        normalize: bool = Field(True, description="Normalize vectors before storage and search")
        n_neighbors: int = Field(64, description="Jasper graph neighbor count")
        alpha: float = Field(1.2, description="Jasper graph alpha")
        workspace_budget: str = Field("10GB", description="Jasper graph build workspace budget")
        beam_width: int = Field(64, description="Jasper search beam width")

        model_config = ConfigDict(arbitrary_types_allowed=True)

    module_name = "mem0.configs.vector_stores.jasper"
    module = sys.modules.get(module_name) or types.ModuleType(module_name)
    module.JasperConfig = JasperConfig
    sys.modules[module_name] = module


def _patch_mem0_vector_config_registry(mem0_vector_config_cls: Any) -> None:
    registry = getattr(mem0_vector_config_cls, "_provider_configs", None)
    if isinstance(registry, dict):
        registry["jasper"] = "JasperConfig"

    private_attrs = getattr(mem0_vector_config_cls, "__private_attributes__", {})
    private_attr = private_attrs.get("_provider_configs") if isinstance(private_attrs, dict) else None
    default = getattr(private_attr, "default", None)
    if isinstance(default, dict):
        default["jasper"] = "JasperConfig"


def _mem0_result_items(results: Any) -> list[Any]:
    if isinstance(results, dict):
        value = results.get("results", [])
        return value if isinstance(value, list) else []
    if isinstance(results, list):
        return results
    return []


def _mem0_item_payload(item: Any) -> dict[str, Any]:
    raw_payload = _item_get(item, "payload")
    if isinstance(raw_payload, dict):
        payload = dict(raw_payload)
    else:
        payload = {}

    memory = _item_get(item, "memory") or payload.get("memory") or payload.get("data") or payload.get("text") or ""
    payload.setdefault("memory", memory)
    payload.setdefault("data", memory)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    item_metadata = _item_get(item, "metadata")
    if isinstance(item_metadata, dict):
        metadata.update(item_metadata)
    if isinstance(item, dict):
        for key, value in item.items():
            if key not in {"id", "memory", "score", "distance", "payload", "metadata"}:
                payload.setdefault(key, value)
                metadata.setdefault(key, value)
    payload["metadata"] = metadata
    return payload


def _item_get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _first_vector(vectors: list[float] | list[list[float]]) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[0] > 0:
        return array[0]
    raise ValueError("vectors must be a one-dimensional vector or a non-empty list of vectors")


def _matches_filters(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if key in {"AND", "$and"}:
            if not isinstance(value, list) or not all(_matches_filters(payload, item) for item in value):
                return False
            continue
        if key in {"OR", "$or"}:
            if not isinstance(value, list) or not any(_matches_filters(payload, item) for item in value):
                return False
            continue
        if key in {"NOT", "$not"}:
            if not isinstance(value, list) or any(_matches_filters(payload, item) for item in value):
                return False
            continue
        if not _matches_value(_payload_value(payload, key), value):
            return False
    return True


def _payload_value(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    if "." in key:
        current: Any = payload
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current
    return None


def _matches_value(actual: Any, expected: Any) -> bool:
    if expected == "*":
        return actual is not None
    if isinstance(expected, dict):
        for operator, value in expected.items():
            if operator == "eq" and actual != value:
                return False
            if operator == "ne" and actual == value:
                return False
            if operator == "in" and actual not in value:
                return False
            if operator == "nin" and actual in value:
                return False
            if operator == "contains" and str(value) not in str(actual):
                return False
            if operator == "icontains" and str(value).lower() not in str(actual).lower():
                return False
            if operator in {"gt", "gte", "lt", "lte"} and not _compare(actual, value, operator):
                return False
        return True
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _compare(actual: Any, expected: Any, operator: str) -> bool:
    try:
        if operator == "gt":
            return actual > expected
        if operator == "gte":
            return actual >= expected
        if operator == "lt":
            return actual < expected
        if operator == "lte":
            return actual <= expected
    except TypeError:
        return False
    return False


def _rerank_hits(hits: list[SearchHit]) -> list[SearchHit]:
    return [
        SearchHit(id=hit.id, payload=hit.payload, score=hit.score, distance=hit.distance, rank=rank)
        for rank, hit in enumerate(hits, start=1)
    ]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _normalize_distance(distance: str) -> str:
    lowered = str(distance).lower()
    if lowered in {"cosine", "ip", "dot"}:
        return "ip"
    if lowered in {"euclidean", "l2"}:
        return "l2"
    return lowered


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
