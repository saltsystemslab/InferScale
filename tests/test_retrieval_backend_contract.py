from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from locomo_jasper_bench.config import BenchmarkConfig, parse_args
from locomo_jasper_bench.retrieval.jasper_vector_store import JasperVectorStore
from locomo_jasper_bench.retrieval.mem0_adapter import (
    Mem0JasperVectorStore,
    _validate_search_hits,
)
from locomo_jasper_bench.retrieval.mem0_provider import (
    _install_jasper_config_module,
    _validate_resolved_backend,
    build_mem0_config,
)
from locomo_jasper_bench.retrieval.memory_builder import _store_config
from locomo_jasper_bench.retrieval.qdrant_vector_store import QdrantVectorStore
from locomo_jasper_bench.vector_types import VECTOR_DISTANCE, SearchHit, VectorStoreConfig


class _FakeStore:
    def __init__(self, root: object, config: VectorStoreConfig) -> None:
        self.root = root
        self.config = config
        self.vector_count = 0
        self.dim = None


def _hit(item_id: str, rank: int, distance: float = 0.5) -> SearchHit:
    return SearchHit(
        id=item_id,
        payload={"memory": item_id},
        score=-distance,
        distance=distance,
        rank=rank,
    )


def test_mem0_pydantic_config_retains_backend_and_enforces_inner_product(tmp_path: object) -> None:
    config = build_mem0_config(
        store_root=tmp_path,
        vector_config=VectorStoreConfig(backend="qdrant"),
        embedding_model="embedding-model",
        embedding_api_key=None,
        embedding_base_url=None,
        memory_llm_model="answer/model",
    )
    _install_jasper_config_module()
    jasper_config = sys.modules["mem0.configs.vector_stores.jasper"].JasperConfig

    parsed = jasper_config(**config["vector_store"]["config"])

    assert parsed.backend == "qdrant"
    assert parsed.model_dump()["backend"] == "qdrant"
    assert parsed.distance == VECTOR_DISTANCE
    assert config["vector_store"]["config"]["distance"] == VECTOR_DISTANCE
    with pytest.raises(ValidationError, match="literal_error"):
        jasper_config(distance="cosine")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        jasper_config(backend="qdrant", omitted_contract_field=True)


def test_adapter_dispatches_exhaustively(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.setattr("locomo_jasper_bench.retrieval.mem0_adapter.JasperVectorStore", _FakeStore)
    monkeypatch.setattr("locomo_jasper_bench.retrieval.mem0_adapter.QdrantVectorStore", _FakeStore)

    jasper = Mem0JasperVectorStore(path=tmp_path, backend="jasper")
    qdrant = Mem0JasperVectorStore(path=tmp_path, backend="qdrant")

    assert jasper.store.config.backend == "jasper"
    assert qdrant.store.config.backend == "qdrant"
    with pytest.raises(ValueError, match="requires distance='ip'"):
        Mem0JasperVectorStore(path=tmp_path, backend="qdrant", distance="cosine")
    with pytest.raises(ValueError, match="Unsupported vector backend"):
        Mem0JasperVectorStore(path=tmp_path, backend="typo")


def test_backend_validation_rejects_concrete_store_mismatch() -> None:
    closed: list[bool] = []
    vector_store = SimpleNamespace(
        config=VectorStoreConfig(backend="qdrant"),
        store=object.__new__(JasperVectorStore),
        close=lambda: closed.append(True),
    )

    with pytest.raises(RuntimeError, match="backend mismatch"):
        _validate_resolved_backend(SimpleNamespace(vector_store=vector_store), requested_backend="qdrant")

    assert closed == [True]


def test_search_contract_rejects_incomplete_duplicate_and_sentinel_hits() -> None:
    with pytest.raises(RuntimeError, match="returned 1 hits, expected 2"):
        _validate_search_hits([_hit("one", 1)], expected_count=2, backend="jasper")
    with pytest.raises(RuntimeError, match="duplicate result ids"):
        _validate_search_hits([_hit("one", 1), _hit("one", 2)], expected_count=2, backend="jasper")
    with pytest.raises(RuntimeError, match="sentinel-like"):
        _validate_search_hits([_hit("one", 1, float(np.finfo(np.float32).max))], expected_count=1, backend="jasper")


def test_jasper_stages_exact_search_and_builds_graph_only_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    store = JasperVectorStore(tmp_path, VectorStoreConfig(backend="jasper", beam_width=1))
    store.add_many(
        [[1.0, 0.0], [0.0, 1.0]],
        [{"memory": "one"}, {"memory": "two"}],
        ["one", "two"],
    )

    builds: list[bool] = []
    monkeypatch.setattr(
        store,
        "_build_jasper_graph",
        lambda: builds.append(True) or object(),
    )

    hits, metrics = store.search([1.0, 0.0], top_k=2)
    store.search([1.0, 0.0], top_k=2)
    assert [hit.id for hit in hits] == ["one", "two"]
    assert metrics.jasper_effective_beam_width == 2
    assert builds == []

    store.finalize()
    store.finalize()
    assert builds == [True]


def test_jasper_graph_build_receives_inner_product(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    graph_builds: list[dict[str, object]] = []

    class FakeTensor:
        def to(self, **_kwargs: object) -> FakeTensor:
            return self

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
            memory_allocated=lambda: 0,
            empty_cache=lambda: None,
        ),
        float32="float32",
        from_numpy=lambda _vectors: FakeTensor(),
    )
    fake_jasper = SimpleNamespace(
        Graph=SimpleNamespace(
            build=lambda _vectors, **kwargs: graph_builds.append(dict(kwargs)) or object()
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "jasper", fake_jasper)
    store = JasperVectorStore(tmp_path, VectorStoreConfig(backend="jasper"))
    store.add_many([[1.0, 0.0]], [{"memory": "one"}], ["one"])

    store._build_jasper_graph()

    assert graph_builds[0]["distance"] == VECTOR_DISTANCE


def test_jasper_all_matching_scope_filter_keeps_top_k_gpu_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    store = JasperVectorStore(tmp_path, VectorStoreConfig(backend="jasper", beam_width=1))
    store.add_many(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        [
            {"memory": "one", "user_id": "sample-1"},
            {"memory": "two", "user_id": "sample-1"},
            {"memory": "three", "user_id": "sample-1"},
        ],
        ["one", "two", "three"],
    )
    store._finalized = True
    store._graph = object()
    searches: list[tuple[int, int]] = []

    def fake_search(query: np.ndarray, top_k: int, *, beam_width: int) -> tuple[list[SearchHit], float]:
        searches.append((top_k, beam_width))
        return store._search_exact(query, top_k)

    monkeypatch.setattr(store, "_search_jasper", fake_search)

    hits, metrics = store.search(
        [1.0, 0.0],
        top_k=2,
        filters={"user_id": "sample-1"},
    )

    assert [hit.id for hit in hits] == ["one", "two"]
    assert searches == [(2, 2)]
    assert metrics.jasper_effective_beam_width == 2


def test_jasper_partial_filter_uses_complete_exact_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    store = JasperVectorStore(tmp_path, VectorStoreConfig(backend="jasper", beam_width=1))
    store.add_many(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        [
            {"memory": "one", "user_id": "sample-1"},
            {"memory": "two", "user_id": "sample-2"},
            {"memory": "three", "user_id": "sample-1"},
        ],
        ["one", "two", "three"],
    )
    store._finalized = True
    store._graph = object()
    monkeypatch.setattr(
        store,
        "_search_jasper",
        lambda *_args, **_kwargs: pytest.fail("partial filters must not use incomplete GPU over-fetch"),
    )

    hits, _ = store.search(
        [1.0, 0.0],
        top_k=2,
        filters={"user_id": "sample-1"},
    )

    assert [hit.id for hit in hits] == ["one", "three"]


def test_effective_beam_expands_to_top_k_and_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCOMO_KV_CONTEXT_WINDOW", raising=False)
    config = parse_args(["--skip-judge", "--top-k", "100", "--jasper-beam-width", "64"])

    assert config.context_window == 0
    assert config.context_window_unit == "turns"
    assert config.context_window_semantics == "encoding-prefix-discard-v1"
    assert config.memory_unit == "mem0-fact"
    assert config.mem0_infer is True
    assert config.jasper_effective_beam_width == 100
    assert config.to_jsonable()["context_window_semantics"] == "encoding-prefix-discard-v1"
    assert config.to_jsonable()["memory_unit"] == "mem0-fact"
    assert config.to_jsonable()["mem0_infer"] is True
    assert config.to_jsonable()["jasper_effective_beam_width"] == 100
    assert _store_config(config).beam_width == 100


def test_qdrant_has_no_effective_jasper_beam() -> None:
    config = BenchmarkConfig(vector_backend="qdrant", top_k=100, jasper_beam_width=64)

    assert config.jasper_effective_beam_width is None
    assert _store_config(config).beam_width == 64


def test_nonzero_turn_context_is_accepted_for_prefix_backend() -> None:
    config = parse_args(
        [
            "--skip-judge",
            "--answer-backend",
            "vllm-prefix",
            "--vector-backend",
            "qdrant",
            "--context-window",
            "1",
        ]
    )

    assert config.context_window == 1
    assert config.answer_backend == "vllm-prefix"


def test_negative_context_window_remains_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--context-window", "-1"])


def test_primary_configuration_has_no_distance_override() -> None:
    assert "vector_distance" not in BenchmarkConfig().to_jsonable()
    with pytest.raises(TypeError, match="vector_distance"):
        BenchmarkConfig(vector_distance="cosine")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="distance"):
        VectorStoreConfig(distance="cosine")  # type: ignore[call-arg]
    with pytest.raises(SystemExit):
        parse_args(["--skip-judge", "--vector-distance", VECTOR_DISTANCE])


def test_local_qdrant_returns_complete_unique_results(tmp_path: object) -> None:
    from qdrant_client import models

    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    store.add_many(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        [{"memory": "one"}, {"memory": "two"}, {"memory": "three"}],
        ["one", "two", "three"],
    )

    hits, metrics = store.search([1.0, 0.0], top_k=2)
    _validate_search_hits(hits, expected_count=2, backend="qdrant")

    collection = store._client.get_collection(collection_name="memories")
    assert collection.config.params.vectors.distance == models.Distance.DOT
    assert [hit.id for hit in hits] == ["one", "three"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].distance == pytest.approx(-1.0)
    assert metrics.vector_backend == "qdrant"
    assert metrics.jasper_effective_beam_width is None
    assert not hasattr(store, "memory_stats")
    store.close()


def test_jasper_and_qdrant_rank_candidates_by_inner_product(tmp_path: object) -> None:
    vectors = [[2.0, 0.0], [0.9, 0.1]]
    payloads = [{"memory": "larger product"}, {"memory": "smaller product"}]
    ids = ["larger", "smaller"]
    stores = (
        JasperVectorStore(tmp_path, VectorStoreConfig(backend="jasper")),
        QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant")),
    )

    try:
        for store in stores:
            store.add_many(vectors, payloads, ids)
            hits, _ = store.search([1.0, 0.0], top_k=2)
            assert [hit.id for hit in hits] == ["larger", "smaller"]
            assert hits[0].score == pytest.approx(2.0)
            assert hits[0].distance == pytest.approx(-2.0)
    finally:
        for store in stores:
            store.close()


def test_local_qdrant_pushes_simple_scope_filter_into_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    store.add_many(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        [
            {"memory": "one", "user_id": "sample-1"},
            {"memory": "two", "user_id": "sample-2"},
            {"memory": "three", "user_id": "sample-1"},
        ],
        ["one", "two", "three"],
    )
    original_query_points = store._client.query_points
    query_calls: list[dict[str, object]] = []

    def recording_query_points(**kwargs: object) -> object:
        query_calls.append(dict(kwargs))
        return original_query_points(**kwargs)

    monkeypatch.setattr(store._client, "query_points", recording_query_points)
    try:
        hits, _ = store.search(
            [1.0, 0.0],
            top_k=2,
            filters={"user_id": "sample-1"},
        )
    finally:
        store.close()

    assert [hit.id for hit in hits] == ["one", "three"]
    assert query_calls[0]["limit"] == 2
    assert query_calls[0]["query_filter"] is not None


def test_local_qdrant_finalize_caches_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant"))
    store.add_many(
        [[1.0, 0.0], [0.0, 1.0]],
        [
            {"data": "Alice likes tea", "user_id": "sample-1"},
            {"data": "Bob likes coffee", "user_id": "sample-1"},
        ],
        ["tea", "coffee"],
    )
    store.finalize()
    monkeypatch.setattr(
        store._client,
        "scroll",
        lambda **_kwargs: pytest.fail("finalized Qdrant rows should be served from memory"),
    )
    try:
        assert [item_id for item_id, _ in store.rows({"user_id": "sample-1"})] == [
            "tea",
            "coffee",
        ]
    finally:
        store.close()


def test_mem0_adapter_supports_filtered_listing_and_entity_crud(tmp_path: object) -> None:
    store = Mem0JasperVectorStore(path=tmp_path, backend="qdrant")
    try:
        store.insert(
            vectors=[[1.0, 0.0], [0.0, 1.0]],
            ids=["tea", "coffee"],
            payloads=[
                {
                    "data": "Alice likes green tea",
                    "user_id": "sample-1",
                },
                {
                    "data": "Bob likes coffee",
                    "user_id": "sample-2",
                },
            ],
        )

        assert [hit.id for hit in store.list(filters={"user_id": "sample-1"})] == ["tea"]

        row = store.get("tea")
        assert row is not None
        next_payload = {**row.payload, "linked_memory_ids": ["fact-1"]}
        store.update("tea", payload=next_payload)
        assert store.get("tea").payload["linked_memory_ids"] == ["fact-1"]  # type: ignore[union-attr]
        store.delete("tea")
        assert store.get("tea") is None
    finally:
        store.close()
