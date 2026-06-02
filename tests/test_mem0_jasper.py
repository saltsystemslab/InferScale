from __future__ import annotations

import numpy as np

from locomo_jasper_bench.mem0_jasper import Mem0JasperVectorStore, build_mem0_config, mem0_results_to_search_hits
from locomo_jasper_bench.jasper_store import VectorStoreConfig


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
