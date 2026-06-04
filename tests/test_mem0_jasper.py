from __future__ import annotations

import numpy as np
import pytest

from locomo_jasper_bench.mem0_jasper import Mem0JasperVectorStore, build_mem0_config, mem0_results_to_search_hits
from locomo_jasper_bench.jasper_store import SearchHit, SearchMetrics, VectorStoreConfig


def test_mem0_jasper_vector_store_insert_search_filter_and_finalize(tmp_path):
    store = Mem0JasperVectorStore(
        collection_name="memories",
        path=str(tmp_path),
        backend="numpy",
        distance="ip",
        normalize=True,
    )
    ids = store.insert(
        vectors=[
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0.9, 0, 0], dtype=np.float32),
        ],
        payloads=[
            {"data": "alpha", "user_id": "u1", "metadata": {"speaker": "Alice"}},
            {"data": "beta", "user_id": "u2", "metadata": {"speaker": "Bob"}},
            {"data": "alpha later", "user_id": "u2", "metadata": {"speaker": "Alice"}},
        ],
        ids=["a", "b", "c"],
    )
    build_metrics = store.finalize()
    hits = store.search("alpha", [1, 0, 0], top_k=2)
    filtered = store.search("alpha", [1, 0, 0], top_k=2, filters={"user_id": "u2"})
    metadata_filtered = store.search("alpha", [1, 0, 0], top_k=2, filters={"speaker": "Alice"})
    store.close()

    assert ids == ["a", "b", "c"]
    assert build_metrics.backend == "numpy"
    assert build_metrics.indexed_vector_count == 3
    assert hits[0].id == "a"
    assert filtered[0].id == "c"
    assert all(hit.payload["user_id"] == "u2" for hit in filtered)
    assert {hit.id for hit in metadata_filtered} == {"a", "c"}


def test_mem0_jasper_insert_normalizes_payload_metadata_without_mutating_input(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path), backend="numpy")
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


def test_mem0_results_to_search_hits_deduplicates_ids_and_turns():
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

    assert [hit.id for hit in hits] == ["mem-1", "mem-3"]
    assert [hit.rank for hit in hits] == [1, 2]


class FakeSearchStore:
    def __init__(self, *, vector_count=50, dim=3, hits_by_k=None):
        self.vector_count = vector_count
        self.dim = dim
        self.calls = []
        self.hits_by_k = hits_by_k or {}

    def search(self, query_vector, top_k):
        self.calls.append(top_k)
        hits = self.hits_by_k.get(top_k) or self.hits_by_k.get("default") or []
        return hits[:top_k], SearchMetrics(
            backend="jasper",
            search_time_ms=1.0,
            indexed_vector_count=self.vector_count,
            embedding_dim=self.dim,
        )


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


def test_mem0_jasper_search_deduplicates_backend_hits(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path), backend="numpy")
    fake_store = FakeSearchStore(
        vector_count=50,
        hits_by_k={
            "default": [
                _hit("a", 1.0, turn_id="t1"),
                _hit("a", 0.9, turn_id="t1"),
                _hit("b", 0.8, turn_id="t1"),
                _hit("c", 0.7, turn_id="t2"),
                _hit("d", 0.6, turn_id="t3"),
            ]
        },
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=3)

    assert [hit.id for hit in hits] == ["a", "c", "d"]
    assert fake_store.calls == [8]
    assert store.last_search_metrics.search_time_ms == 1.0


def test_mem0_jasper_search_uses_bounded_filter_overfetch(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path), backend="numpy")
    fake_store = FakeSearchStore(
        vector_count=50,
        hits_by_k={
            "default": [
                _hit("a", 1.0, user_id="u2", turn_id="t1"),
                _hit("b", 0.9, user_id="u2", turn_id="t2"),
                _hit("c", 0.8, user_id="u1", turn_id="t3"),
            ]
        },
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=2, filters={"user_id": "u2"})

    assert [hit.id for hit in hits] == ["a", "b"]
    assert fake_store.calls == [12]
    assert fake_store.calls[0] < fake_store.vector_count


def test_mem0_jasper_search_expands_when_filter_is_restrictive(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path), backend="numpy")
    fake_store = FakeSearchStore(
        vector_count=50,
        hits_by_k={
            12: [
                _hit("a", 1.0, user_id="u1", turn_id="t1"),
                _hit("b", 0.9, user_id="u2", turn_id="t2"),
            ],
            24: [
                _hit("a", 1.0, user_id="u1", turn_id="t1"),
                _hit("b", 0.9, user_id="u2", turn_id="t2"),
                _hit("c", 0.8, user_id="u2", turn_id="t3"),
            ],
        },
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=2, filters={"user_id": "u2"})

    assert [hit.id for hit in hits] == ["b", "c"]
    assert fake_store.calls == [12, 24]
    assert store.last_search_metrics.search_time_ms == 2.0


def test_mem0_jasper_broad_user_filter_uses_requested_top_k(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path / "conv-26"), backend="jasper", beam_width=64)
    fake_store = FakeSearchStore(
        vector_count=419,
        hits_by_k={
            "default": [
                _hit(f"mem-{index}", 1.0 - index * 0.01, user_id="conv-26", turn_id=f"t{index}")
                for index in range(20)
            ]
        },
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=20, filters={"user_id": "conv-26"})

    assert len(hits) == 20
    assert fake_store.calls == [20]


def test_mem0_jasper_scoped_broad_user_filter_does_not_drop_payloads_without_user_id(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path / "conv-26"), backend="jasper", beam_width=64)
    fake_store = FakeSearchStore(
        vector_count=419,
        hits_by_k={
            "default": [
                _hit(f"mem-{index}", 1.0 - index * 0.01, user_id=None, turn_id=f"t{index}")
                for index in range(3)
            ]
        },
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=3, filters={"user_id": "conv-26"})

    assert [hit.id for hit in hits] == ["mem-0", "mem-1", "mem-2"]
    assert fake_store.calls == [3]


def test_mem0_jasper_mismatched_broad_user_filter_stays_strict(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path / "conv-25"), backend="jasper", beam_width=64)
    fake_store = FakeSearchStore(
        vector_count=64,
        hits_by_k={
            "default": [
                _hit(f"mem-{index}", 1.0 - index * 0.01, user_id=None, turn_id=f"t{index}")
                for index in range(3)
            ]
        },
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=2, filters={"user_id": "conv-26"})

    assert hits == []
    assert fake_store.calls == [12, 24, 48, 64]


def test_mem0_jasper_restrictive_filter_never_exceeds_beam_width(tmp_path):
    store = Mem0JasperVectorStore(path=str(tmp_path), backend="jasper", beam_width=64)
    fake_store = FakeSearchStore(
        vector_count=419,
        hits_by_k={
            20: [
                _hit("a", 1.0, user_id="u1", turn_id="t1"),
                _hit("b", 0.9, user_id="u2", turn_id="t2"),
            ],
            40: [
                _hit("a", 1.0, user_id="u1", turn_id="t1"),
                _hit("b", 0.9, user_id="u2", turn_id="t2"),
                _hit("c", 0.8, user_id="u2", turn_id="t3"),
            ],
            64: [
                _hit("a", 1.0, user_id="u1", turn_id="t1"),
                _hit("b", 0.9, user_id="u2", turn_id="t2"),
                _hit("c", 0.8, user_id="u2", turn_id="t3"),
                _hit("d", 0.7, user_id="u2", turn_id="t4"),
            ],
        },
    )
    store.store = fake_store

    hits = store.search("query", [1, 0, 0], top_k=5, filters={"metadata.turn_id": ["t2", "t3", "t4"]})

    assert [hit.id for hit in hits] == ["b", "c", "d"]
    assert fake_store.calls == [20, 40, 64]
    assert max(fake_store.calls) == 64


def test_build_mem0_config_uses_jasper_provider_without_qdrant(tmp_path):
    config = build_mem0_config(
        store_root=tmp_path,
        vector_config=VectorStoreConfig(backend="numpy", distance="ip", normalize=True),
        embedding_model="text-embedding-3-small",
        embedding_api_key="key",
        embedding_base_url="http://embeddings.test/v1",
    )

    assert config["vector_store"]["provider"] == "jasper"
    assert config["vector_store"]["config"]["backend"] == "numpy"
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["openai_base_url"] == "http://embeddings.test/v1"


def test_mem0_qdrant_vector_store_insert_search_filter_and_mutate(tmp_path):
    pytest.importorskip("qdrant_client")
    store = Mem0JasperVectorStore(
        collection_name="memories",
        path=str(tmp_path),
        backend="qdrant",
        distance="ip",
        normalize=True,
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
    build_metrics = store.finalize()
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
    assert build_metrics.backend == "qdrant"
    assert build_metrics.indexed_vector_count == 3
    assert {hit.id for hit in hits} == {"a", "c"}
    assert filtered[0].id == "c"
    assert updated is not None
    assert updated.payload["data"] == "beta updated"
    assert {item.id for item in listed} == {"a", "c"}
    assert after_reset == []
