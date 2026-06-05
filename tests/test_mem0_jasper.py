from __future__ import annotations

import numpy as np
import pytest

from locomo_jasper_bench.jasper_store import SearchHit, SearchMetrics, VectorStoreConfig
from locomo_jasper_bench.mem0_jasper import Mem0JasperVectorStore, build_mem0_config, mem0_results_to_search_hits


class FakeSearchStore:
    def __init__(self, hits, *, vector_count=50, dim=3):
        self.hits = hits
        self.vector_count = vector_count
        self.dim = dim
        self.calls = []

    def search(self, query_vector, top_k):
        self.calls.append(top_k)
        return self.hits[:top_k], SearchMetrics(search_time_ms=1.0)


def _hit(item_id, score, *, user_id="u1", turn_id=None):
    metadata = {}
    if turn_id is not None:
        metadata["turn_id"] = turn_id
    payload = {"data": item_id, "metadata": metadata}
    if user_id is not None:
        payload["user_id"] = user_id
    return SearchHit(
        id=item_id,
        payload=payload,
        score=score,
        distance=score,
        rank=0,
    )


def test_mem0_jasper_insert_normalizes_payload_metadata_without_mutating_input(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path), backend="jasper")
    payloads = [
        {"data": "alpha", "metadata": {"user_id": "u1", "turn_id": "t1"}},
        {"data": "beta", "user_id": "u2", "turn_id": "t2", "metadata": {}},
    ]

    store.insert(
        vectors=[
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
        ],
        payloads=payloads,
        ids=["a", "b"],
    )
    listed = {hit.id: hit.payload for hit in store.list(limit=10)}
    store.close()

    assert listed["a"]["user_id"] == "u1"
    assert listed["a"]["metadata"]["user_id"] == "u1"
    assert listed["a"]["turn_id"] == "t1"
    assert listed["b"]["user_id"] == "u2"
    assert listed["b"]["metadata"]["user_id"] == "u2"
    assert listed["b"]["metadata"]["turn_id"] == "t2"
    assert "user_id" not in payloads[0]
    assert "user_id" not in payloads[1]["metadata"]


def test_mem0_results_to_search_hits_normalizes_wrapped_results():
    hits = mem0_results_to_search_hits(
        {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "Alice: adopted Pixel.",
                    "score": 0.7,
                    "metadata": {"turn_id": "t1"},
                }
            ]
        }
    )

    assert hits[0].id == "mem-1"
    assert hits[0].rank == 1
    assert hits[0].payload["memory"] == "Alice: adopted Pixel."
    assert hits[0].payload["metadata"]["turn_id"] == "t1"


def test_mem0_results_to_search_hits_preserves_duplicates_and_order():
    hits = mem0_results_to_search_hits(
        {
            "results": [
                {"id": "mem-1", "memory": "one", "score": 0.9, "metadata": {"turn_id": "t1"}},
                {"id": "mem-1", "memory": "one duplicate", "score": 0.8, "metadata": {"turn_id": "t1"}},
                {"id": "mem-2", "memory": "turn duplicate", "score": 0.7, "metadata": {"turn_id": "t1"}},
                {"id": "mem-3", "memory": "two", "score": 0.6, "metadata": {"turn_id": "t2"}},
            ]
        }
    )

    assert [hit.id for hit in hits] == ["mem-1", "mem-1", "mem-2", "mem-3"]
    assert [hit.rank for hit in hits] == [1, 2, 3, 4]


def test_mem0_jasper_search_preserves_backend_order_and_duplicates(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path), backend="jasper")
    fake_store = FakeSearchStore(
        [
            _hit("a", 1.0, turn_id="t1"),
            _hit("a", 0.9, turn_id="t1"),
            _hit("b", 0.8, turn_id="t1"),
            _hit("c", 0.7, turn_id="t2"),
        ]
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=4)

    assert [hit.id for hit in hits] == ["a", "a", "b", "c"]
    assert fake_store.calls == [4]
    assert store.last_search_metrics.search_time_ms == 1.0


def test_mem0_jasper_search_filters_once_after_backend_search(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path), backend="jasper")
    fake_store = FakeSearchStore(
        [
            _hit("a", 1.0, user_id="u1", turn_id="t1"),
            _hit("b", 0.9, user_id="u2", turn_id="t2"),
            _hit("c", 0.8, user_id="u2", turn_id="t3"),
        ]
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=3, filters={"user_id": "u2"})

    assert [hit.id for hit in hits] == ["b", "c"]
    assert fake_store.calls == [3]


def test_mem0_jasper_scoped_broad_user_filter_does_not_drop_payloads_without_user_id(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path / "conv-26"), backend="jasper")
    fake_store = FakeSearchStore(
        [
            _hit(f"mem-{index}", 1.0 - index * 0.01, user_id=None, turn_id=f"t{index}")
            for index in range(3)
        ],
        vector_count=419,
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=3, filters={"user_id": "conv-26"})

    assert [hit.id for hit in hits] == ["mem-0", "mem-1", "mem-2"]
    assert fake_store.calls == [3]


def test_mem0_jasper_mismatched_broad_user_filter_stays_strict(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path / "conv-25"), backend="jasper")
    fake_store = FakeSearchStore(
        [
            _hit(f"mem-{index}", 1.0 - index * 0.01, user_id=None, turn_id=f"t{index}")
            for index in range(3)
        ],
        vector_count=64,
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=2, filters={"user_id": "conv-26"})

    assert hits == []
    assert fake_store.calls == [2]


def test_build_mem0_config_uses_jasper_provider_without_qdrant(tmp_path):
    config = build_mem0_config(
        store_root=tmp_path,
        vector_config=VectorStoreConfig(backend="jasper", distance="ip"),
        embedding_model="text-embedding-3-small",
        embedding_api_key="key",
        embedding_base_url="http://embeddings.test/v1",
    )

    assert config["vector_store"]["provider"] == "jasper"
    assert config["vector_store"]["config"]["backend"] == "jasper"
    assert config["vector_store"]["config"]["alpha"] == 1.0
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["openai_base_url"] == "http://embeddings.test/v1"


def test_mem0_qdrant_vector_store_insert_search_filter_and_mutate(tmp_path):
    pytest.importorskip("qdrant_client")
    store = Mem0JasperVectorStore(
        collection_name="memories",
        path=str(tmp_path),
        backend="qdrant",
        distance="ip",
    )
    ids = store.insert(
        vectors=[
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0.9, 0, 0], dtype=np.float32),
        ],
        payloads=[
            {"data": "alpha", "user_id": "u1", "metadata": {"turn_id": "t1"}},
            {"data": "beta", "user_id": "u2", "metadata": {"turn_id": "t2"}},
            {"data": "alpha later", "user_id": "u2", "metadata": {"turn_id": "t3"}},
        ],
        ids=["a", "b", "c"],
    )
    store.finalize()
    hits = store.search("alpha", [1, 0, 0], top_k=2)
    filtered = store.search("alpha", [1, 0, 0], top_k=2, filters={"user_id": "u2"})
    store.update("b", payload={"data": "beta updated", "user_id": "u2", "metadata": {"turn_id": "t2"}})
    updated = store.get("b")
    store.delete("b")
    listed = store.list(limit=10)
    store.reset()
    after_reset = store.list(limit=10)
    store.close()

    assert ids == ["a", "b", "c"]
    assert {hit.id for hit in hits} == {"a", "c"}
    assert filtered[0].id == "c"
    assert updated is not None
    assert updated.payload["data"] == "beta updated"
    assert {item.id for item in listed} == {"a", "c"}
    assert after_reset == []
