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
from locomo_jasper_bench.vector_types import SearchHit, VectorStoreConfig


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


def test_mem0_pydantic_config_retains_backend_and_rejects_unknown_fields(tmp_path: object) -> None:
    config = build_mem0_config(
        store_root=tmp_path,
        vector_config=VectorStoreConfig(backend="qdrant"),
        embedding_model="embedding-model",
        embedding_api_key=None,
        embedding_base_url=None,
    )
    _install_jasper_config_module()
    jasper_config = sys.modules["mem0.configs.vector_stores.jasper"].JasperConfig

    parsed = jasper_config(**config["vector_store"]["config"])

    assert parsed.backend == "qdrant"
    assert parsed.model_dump()["backend"] == "qdrant"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        jasper_config(backend="qdrant", omitted_contract_field=True)


def test_adapter_dispatches_exhaustively(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.setattr("locomo_jasper_bench.retrieval.mem0_adapter.JasperVectorStore", _FakeStore)
    monkeypatch.setattr("locomo_jasper_bench.retrieval.mem0_adapter.QdrantVectorStore", _FakeStore)

    jasper = Mem0JasperVectorStore(path=tmp_path, backend="jasper")
    qdrant = Mem0JasperVectorStore(path=tmp_path, backend="qdrant")

    assert jasper.store.config.backend == "jasper"
    assert qdrant.store.config.backend == "qdrant"
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


def test_jasper_store_rejects_top_k_larger_than_effective_beam(tmp_path: object) -> None:
    store = JasperVectorStore(tmp_path, VectorStoreConfig(backend="jasper", beam_width=1))
    store.add_many(
        [[1.0, 0.0], [0.0, 1.0]],
        [{"memory": "one"}, {"memory": "two"}],
        ["one", "two"],
    )

    with pytest.raises(ValueError, match="exceeds beam_width"):
        store.search([1.0, 0.0], top_k=2)


def test_effective_beam_expands_to_top_k_and_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCOMO_KV_CONTEXT_WINDOW", raising=False)
    config = parse_args(["--skip-judge", "--top-k", "100", "--jasper-beam-width", "64"])

    assert config.context_window == 0
    assert config.context_window_unit == "sessions"
    assert config.jasper_effective_beam_width == 100
    assert config.to_jsonable()["jasper_effective_beam_width"] == 100
    assert _store_config(config).beam_width == 100


def test_qdrant_has_no_effective_jasper_beam() -> None:
    config = BenchmarkConfig(vector_backend="qdrant", top_k=100, jasper_beam_width=64)

    assert config.jasper_effective_beam_width is None
    assert _store_config(config).beam_width == 64


def test_nonzero_context_is_rejected_for_prefix_backend() -> None:
    with pytest.raises(SystemExit):
        parse_args(
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


def test_local_qdrant_returns_complete_unique_results(tmp_path: object) -> None:
    store = QdrantVectorStore(tmp_path, VectorStoreConfig(backend="qdrant", distance="ip"))
    store.add_many(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        [{"memory": "one"}, {"memory": "two"}, {"memory": "three"}],
        ["one", "two", "three"],
    )

    hits, metrics = store.search([1.0, 0.0], top_k=2)
    _validate_search_hits(hits, expected_count=2, backend="qdrant")

    assert [hit.id for hit in hits] == ["one", "three"]
    assert metrics.vector_backend == "qdrant"
    assert metrics.jasper_effective_beam_width is None
    assert not hasattr(store, "memory_stats")
    store.close()
