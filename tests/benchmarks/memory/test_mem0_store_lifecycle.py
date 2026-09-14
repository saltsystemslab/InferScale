from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.memory.mem0.memory_builder import SampleMemoryBuilder
from benchmarks.memory.mem0.provider import close_mem0_stores
from memory_config import make_memory_config, memory_runtime


def test_cleanup_attempts_both_stores_when_one_delete_fails() -> None:
    closed = []

    def close_entities() -> None:
        closed.append("entities")
        raise RuntimeError("entity collection deletion failed")

    memory = SimpleNamespace(
        vector_store=SimpleNamespace(close=lambda: closed.append("memories")),
        _entity_store=SimpleNamespace(close=close_entities),
    )

    with pytest.raises(RuntimeError, match="entity collection deletion failed"):
        close_mem0_stores(memory)

    assert set(closed) == {"memories", "entities"}


@pytest.mark.parametrize("stage", ["_install_embedding_cache", "_reset_vector_store", "_load_facts_into_memory", "_finalize"])
def test_failed_sample_setup_closes_all_server_stores(tmp_path, monkeypatch, stage) -> None:
    config = make_memory_config(
        vector_backend="qdrant",
        runtime=memory_runtime(storage={"cache_root": str(tmp_path)}),
    )
    builder = SampleMemoryBuilder(config)
    closed = []
    memory = SimpleNamespace(
        vector_store=SimpleNamespace(
            config=SimpleNamespace(backend="qdrant"),
            close=lambda: closed.append("memories"),
        ),
        _entity_store=SimpleNamespace(close=lambda: closed.append("entities")),
    )
    monkeypatch.setattr(builder, "load_fact_catalog", lambda _: ())
    monkeypatch.setattr(builder, "_create_memory", lambda *args, **kwargs: memory)
    for method in ("_install_embedding_cache", "_reset_vector_store", "_load_facts_into_memory", "_finalize"):
        monkeypatch.setattr(builder, method, lambda *args: None)

    def fail(*args) -> None:
        raise RuntimeError("sample setup failed")

    monkeypatch.setattr(builder, stage, fail)
    sample = SimpleNamespace(sample_id="sample-1", turns=[])

    with pytest.raises(RuntimeError, match="sample setup failed"):
        builder.build_with_metrics(sample)

    assert set(closed) == {"memories", "entities"}
