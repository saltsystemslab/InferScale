from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from benchmarks.common.vector_types import SearchHit, VectorStoreConfig
from benchmarks.memory import qdrant_index
from benchmarks.memory.qdrant_index import QdrantVectorStore

VECTORS = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.9, 0.1]]
PAYLOADS = [
    {"memory": "one", "user_id": "u1", "n": 1},
    {"memory": "two", "user_id": "u2", "n": 2},
    {"memory": "three", "user_id": "u1", "n": 3},
    {"memory": "four", "user_id": "u3", "n": 2},
]
IDS = ["one", "two", "three", "four"]
QUERY = [1.0, 0.2]
QUERY_POINTS_MS = 7
COUNT_MS = 3


class Clock:
    def __init__(self) -> None:
        self.now_s = 100.0

    def perf_counter(self) -> float:
        return self.now_s

    def advance_ms(self, milliseconds: float) -> None:
        self.now_s += milliseconds / 1000


class ClientCalls:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, store: QdrantVectorStore) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.clock: Clock | None = None
        for name in ("query_points", "count", "scroll"):
            monkeypatch.setattr(
                store._client, name, self._recording(name, getattr(store._client, name))
            )

    def _recording(self, name: str, method: Any) -> Any:
        def call(**arguments: Any) -> Any:
            self.calls.append((name, arguments))
            if self.clock is not None:
                self.clock.advance_ms(QUERY_POINTS_MS if name == "query_points" else COUNT_MS)
            return method(**arguments)

        return call

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def query(self) -> dict[str, Any]:
        [arguments] = [arguments for name, arguments in self.calls if name == "query_points"]
        return arguments


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QdrantVectorStore]:
    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    store.add_many(VECTORS, PAYLOADS, IDS)
    yield store
    store.close()


def _ids(hits: list[SearchHit]) -> list[str]:
    return [hit.id for hit in hits]


def _row_ids(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [item_id for item_id, _ in rows]


@pytest.mark.parametrize(
    "filters, expected_ids",
    [
        ({"n": 2.0}, ["four", "two"]),
        ({"user_id": "u1", "n": {"ne": 2.0}}, ["one", "three"]),
    ],
)
def test_filter_qdrant_cannot_evaluate_is_applied_after_fetching_every_vector(
    store: QdrantVectorStore,
    monkeypatch: pytest.MonkeyPatch,
    filters: dict[str, Any],
    expected_ids: list[str],
) -> None:
    client = ClientCalls(monkeypatch, store)

    hits, _ = store.search(QUERY, top_k=3, filters=filters)

    assert _ids(hits) == expected_ids
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    assert client.query()["limit"] == len(VECTORS)
    assert "query_filter" not in client.query()


def test_search_time_is_the_qdrant_query_alone(
    store: QdrantVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = ClientCalls(monkeypatch, store)
    client.clock = Clock()
    monkeypatch.setattr(qdrant_index, "time", client.clock)

    _, metrics = store.search(QUERY, top_k=2, filters={"user_id": "u1"})

    assert client.names() == ["count", "query_points"]
    assert metrics.search_time_ms == pytest.approx(QUERY_POINTS_MS)


def test_finalized_store_counts_and_lists_rows_without_asking_qdrant(
    store: QdrantVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.finalize()
    client = ClientCalls(monkeypatch, store)

    assert store.count() == 4
    assert store.count({"user_id": "u1"}) == 2
    assert store.count({"n": {"ne": 2.0}}) == 2
    assert _row_ids(store.rows({"n": 2})) == ["four", "two"]
    store.finalize()

    assert client.calls == []
