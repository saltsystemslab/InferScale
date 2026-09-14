from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qdrant_client import models

from benchmarks.common.qdrant_config import QdrantConfig
from benchmarks.common.vector_types import VectorStoreConfig
from benchmarks.memory.qdrant_index import QdrantVectorStore


class FakeQdrantClient:
    """A public-client test double that serializes payloads across its boundary."""

    def __init__(self, collections: dict[str, dict[str, Any]], **kwargs: Any) -> None:
        self.collections = collections
        self.options = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(collections=list(self.collections))

    def create_collection(self, **kwargs: Any) -> None:
        self.calls.append(("create_collection", kwargs))
        self.collections[kwargs["collection_name"]] = {}

    def delete_collection(self, **kwargs: Any) -> None:
        self.calls.append(("delete_collection", kwargs))
        del self.collections[kwargs["collection_name"]]

    def count(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("count", kwargs))
        return SimpleNamespace(count=len(self._points(kwargs)))

    def upsert(self, **kwargs: Any) -> None:
        self.calls.append(("upsert", kwargs))
        points = self.collections[kwargs["collection_name"]]
        for point in kwargs["points"]:
            points[point.id] = json.loads(point.model_dump_json())

    def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("query_points", kwargs))
        points = self._points(kwargs)
        scored = [
            (float(np.dot(point["vector"], kwargs["query"])), point)
            for point in points
        ]
        scored.sort(key=lambda row: (-row[0], row[1]["id"]))
        return SimpleNamespace(
            points=[self._response(point, score) for score, point in scored[:kwargs["limit"]]]
        )

    def scroll(self, **kwargs: Any) -> tuple[list[SimpleNamespace], int | None]:
        self.calls.append(("scroll", kwargs))
        start = kwargs["offset"] or 0
        points = self._points(kwargs)
        end = start + kwargs["limit"]
        return (
            [self._response(point) for point in points[start:end]],
            end if end < len(points) else None,
        )

    def retrieve(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(("retrieve", kwargs))
        points = self.collections[kwargs["collection_name"]]
        return [self._response(points[key]) for key in kwargs["ids"] if key in points]

    def set_payload(self, **kwargs: Any) -> None:
        self.calls.append(("set_payload", kwargs))
        points = self.collections[kwargs["collection_name"]]
        for key in kwargs["points"]:
            if key in points:
                points[key]["payload"].update(json.loads(json.dumps(kwargs["payload"])))

    def update_vectors(self, **kwargs: Any) -> None:
        self.calls.append(("update_vectors", kwargs))
        points = self.collections[kwargs["collection_name"]]
        for point in kwargs["points"]:
            points[point.id]["vector"] = point.vector

    def delete(self, **kwargs: Any) -> None:
        self.calls.append(("delete", kwargs))
        points = self.collections[kwargs["collection_name"]]
        for key in kwargs["points_selector"].points:
            points.pop(key, None)

    def close(self) -> None:
        self.closed = True

    def _points(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        points = list(self.collections[kwargs["collection_name"]].values())
        query_filter = kwargs.get("query_filter") or kwargs.get("count_filter")
        if query_filter is None:
            return points

        def matches(condition: Any, payload: dict[str, Any]) -> bool:
            value = payload.get(condition.key)
            if isinstance(condition.match, models.MatchValue):
                return value == condition.match.value
            return value in condition.match.any

        return [
            point
            for point in points
            if all(matches(condition, point["payload"]) for condition in query_filter.must or [])
            and not any(
                matches(condition, point["payload"]) for condition in query_filter.must_not or []
            )
        ]

    @staticmethod
    def _response(point: dict[str, Any], score: float = 0.0) -> SimpleNamespace:
        return SimpleNamespace(
            id=point["id"],
            score=score,
            payload=json.loads(json.dumps(point["payload"])),
        )


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    server = SimpleNamespace(collections={}, clients=[])

    def create_client(**kwargs: Any) -> FakeQdrantClient:
        client = FakeQdrantClient(server.collections, **kwargs)
        server.clients.append(client)
        return client

    monkeypatch.setattr("qdrant_client.QdrantClient", create_client)
    return server


@pytest.fixture
def store(tmp_path: Path, fake_server: SimpleNamespace) -> QdrantVectorStore:
    result = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    yield result
    result.close()


def test_connects_to_configured_server_and_owns_unique_collections(
    tmp_path: Path, fake_server: SimpleNamespace,
) -> None:
    config = VectorStoreConfig(
        backend="qdrant",
        qdrant=QdrantConfig(url="http://qdrant:7333", grpc_port=7334, timeout=12),
    )
    first = QdrantVectorStore(tmp_path / "sample one", config)
    second = QdrantVectorStore(tmp_path / "sample one", config)
    try:
        assert first._client.options == {
            "url": "http://qdrant:7333", "grpc_port": 7334, "prefer_grpc": True, "timeout": 12,
        }
        assert first._collection_name.startswith("benchmark_jasper_sample_one_")
        assert first._collection_name != second._collection_name
        for current in (first, second):
            current.add_many([[1.0, 0.0]], [{"data": "fact"}], ["same-id"])
        first.close()
        assert second.get("same-id") is not None
        assert first._collection_name not in fake_server.collections
        assert second._collection_name in fake_server.collections
        first.close()
    finally:
        first.close()
        second.close()
    assert fake_server.collections == {}
    assert all(client.closed for client in fake_server.clients)


def test_empty_store_does_not_query_nonexistent_collection(store: QdrantVectorStore) -> None:
    assert store.count() == 0
    assert store.rows() == []
    assert store.get("missing") is None
    assert store.search([1.0, 0.0], 10)[0] == []
    store.update("missing", payload={"data": "unused"})
    store.delete("missing")
    assert store._client.calls == []


def test_search_preserves_original_ids_dot_scores_and_payload_isolation(
    store: QdrantVectorStore,
) -> None:
    payload = {"data": "fact", "linked_memory_ids": ["a"]}
    store.add_many([[2.0, 0.0], [0.9, 0.1]], [payload, {"data": "other"}], ["text-id", "other"])
    payload["linked_memory_ids"].append("mutated after upload")
    hits, metrics = store.search([1.0, 0.0], 2)
    assert [hit.id for hit in hits] == ["text-id", "other"]
    assert [hit.score for hit in hits] == pytest.approx([2.0, 0.9])
    assert [hit.distance for hit in hits] == pytest.approx([-2.0, -0.9])
    assert [hit.rank for hit in hits] == [1, 2]
    assert metrics.vector_backend == "qdrant"
    assert hits[0].payload["linked_memory_ids"] == ["a"]
    assert store._ID_PAYLOAD_KEY not in hits[0].payload
    hits[0].payload["linked_memory_ids"].append("mutated response")
    assert store.get("text-id").payload["linked_memory_ids"] == ["a"]
    create = next(kwargs for name, kwargs in store._client.calls if name == "create_collection")
    query = next(kwargs for name, kwargs in store._client.calls if name == "query_points")
    assert create["vectors_config"].distance == models.Distance.DOT
    assert query["search_params"].exact is True
    assert query["with_payload"] is True


@pytest.mark.parametrize(
    ("filters", "expected", "native"),
    [
        ({"user_id": "a"}, ["one", "three"], True),
        ({"user_id": {"in": ["a", "b"], "ne": "b"}}, ["one", "three"], True),
        ({"weight": {"eq": 0.5}}, ["two", "three"], False),
    ],
)
def test_native_and_fallback_filters_preserve_top_k(
    store: QdrantVectorStore, filters: dict[str, Any], expected: list[str], native: bool,
) -> None:
    store.add_many(
        [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        [
            {"user_id": "a", "weight": 0.2},
            {"user_id": "b", "weight": 0.5},
            {"user_id": "a", "weight": 0.5},
        ],
        ["one", "two", "three"],
    )
    hits, _ = store.search([1.0, 0.0], 2, filters)
    assert [hit.id for hit in hits] == expected
    assert [hit.rank for hit in hits] == [1, 2]
    query = next(kwargs for name, kwargs in store._client.calls if name == "query_points")
    assert (query.get("query_filter") is not None) is native
    assert query["limit"] == (2 if native else 3)


def test_finalize_paginates_and_avoids_count_rpcs_on_search(store: QdrantVectorStore) -> None:
    size = 2 * store._SCROLL_PAGE_SIZE + 3
    store.add_many(
        [[float(index), 1.0] for index in range(size)],
        [{"user_id": str(index % 2)} for index in range(size)],
        [str(index) for index in range(size)],
    )
    store.finalize()
    scrolls = [kwargs for name, kwargs in store._client.calls if name == "scroll"]
    assert len(scrolls) == 3
    assert [kwargs["offset"] for kwargs in scrolls] == [None, 256, 512]
    store._client.calls.clear()
    assert store.vector_count == size
    assert store.count({"user_id": "0"}) == (size + 1) // 2
    assert len(store.rows()) == size
    store.search([1.0, 0.0], 10)
    store.search([1.0, 0.0], 10)
    assert [name for name, _ in store._client.calls] == ["query_points", "query_points"]


def test_writes_wait_and_invalidate_finalized_caches(store: QdrantVectorStore) -> None:
    store.add_many([[1.0, 0.0]], [{"user_id": "a"}], ["one"])
    store.finalize()
    store.add_many([[0.0, 1.0]], [{"user_id": "b"}], ["two"])
    assert store.vector_count == 2
    store.finalize()
    store.update("two", vector=[2.0, 0.0], payload={"user_id": "a"})
    assert store.count({"user_id": "a"}) == 2
    assert store.search([1.0, 0.0], 1)[0][0].id == "two"
    store.finalize()
    store.delete("one")
    assert store.vector_count == 1
    assert store.get("one") is None
    assert [item_id for item_id, _ in store.rows()] == ["two"]
    writes = {"upsert", "set_payload", "update_vectors", "delete"}
    assert all(kwargs["wait"] is True for name, kwargs in store._client.calls if name in writes)


def test_reset_removes_collection_and_allows_a_new_dimension(store: QdrantVectorStore) -> None:
    store.add_many([[1.0, 0.0]], [{}], ["one"])
    store.finalize()
    store.reset()
    assert store.vector_count == 0
    assert store.dim is None
    store.add_many([[1.0, 0.0, 0.0]], [{}], ["one"])
    assert store.dim == 3
    assert store.vector_count == 1


def test_server_failure_is_not_reported_as_empty_store(
    store: QdrantVectorStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store.add_many([[1.0, 0.0]], [{}], ["one"])

    def fail(**kwargs: Any) -> None:
        raise ConnectionError("connection lost")

    monkeypatch.setattr(store._client, "count", fail)
    with pytest.raises(ConnectionError, match="connection lost"):
        _ = store.vector_count


def test_connection_failure_closes_client_and_explains_server_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_server: SimpleNamespace,
) -> None:
    def fail(self: FakeQdrantClient) -> None:
        raise ConnectionError("refused")

    monkeypatch.setattr(FakeQdrantClient, "get_collections", fail)
    with pytest.raises(RuntimeError, match="Cannot connect to the Qdrant server") as error:
        QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    assert isinstance(error.value.__cause__, ConnectionError)
    assert fake_server.clients[0].closed


def test_close_reports_cleanup_failure_but_closes_client(
    tmp_path: Path, fake_server: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    store.add_many([[1.0, 0.0]], [{}], ["one"])

    def fail(**kwargs: Any) -> None:
        raise ConnectionError("cleanup failed")

    monkeypatch.setattr(store._client, "delete_collection", fail)
    with pytest.raises(ConnectionError, match="cleanup failed"):
        store.close()
    assert store._client.closed
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.search([1.0, 0.0], 1)


def test_query_can_explicitly_enable_approximate_search(
    tmp_path: Path, fake_server: SimpleNamespace,
) -> None:
    store = QdrantVectorStore(
        tmp_path,
        VectorStoreConfig(backend="qdrant", qdrant=QdrantConfig(exact=False)),
    )
    try:
        store.add_many([[1.0, 0.0]], [{}], ["one"])
        store.search([1.0, 0.0], 1)
        query = next(kwargs for name, kwargs in store._client.calls if name == "query_points")
        assert query["search_params"].exact is False
    finally:
        store.close()


def test_real_server_exact_scores_filters_pagination_and_cleanup(
    tmp_path: Path, qdrant_server_config: VectorStoreConfig,
) -> None:
    store = QdrantVectorStore(tmp_path, qdrant_server_config)
    store._SCROLL_PAGE_SIZE = 2
    try:
        store.add_many(
            [[2.0, 0.0], [0.9, 0.1], [0.1, 0.9]],
            [
                {"user_id": "a", "weight": 0.5, "bucket": 1, "linked_memory_ids": ["fact-1"]},
                {"user_id": "b", "weight": 0.5, "bucket": 2, "linked_memory_ids": ["fact-2"]},
                {"user_id": "a", "weight": 0.2, "bucket": 3},
            ],
            ["one", "two", "three"],
        )
        store.finalize()
        assert store.vector_count == 3
        assert {key for key, _ in store.rows()} == {"one", "two", "three"}
        hits, _ = store.search([1.0, 0.0], 2, {"user_id": "a"})
        assert [hit.id for hit in hits] == ["one", "three"]
        assert [hit.score for hit in hits] == pytest.approx([2.0, 0.1])
        hits[0].payload["linked_memory_ids"].append("changed client-side")
        assert store.get("one").payload["linked_memory_ids"] == ["fact-1"]
        fallback, _ = store.search([1.0, 0.0], 2, {"weight": 0.5})
        assert [hit.id for hit in fallback] == ["one", "two"]
        integer_hits, _ = store.search([1.0, 0.0], 2, {"bucket": {"in": [1, 3]}})
        assert [hit.id for hit in integer_hits] == ["one", "three"]
        array_hits, _ = store.search(
            [1.0, 0.0], 2, {"linked_memory_ids": {"eq": ["fact-2"]}},
        )
        assert [hit.id for hit in array_hits] == ["two"]
        store.update("two", payload={"user_id": "a"})
        assert store.count({"user_id": "a"}) == 3
        store.delete("three")
        assert store.vector_count == 2
        store.reset()
        assert not store._client.collection_exists(store._collection_name)
    finally:
        store.close()
