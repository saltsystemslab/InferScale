from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.mem0.provider import Mem0Settings
from benchmarks.memory.throughput import stores
from benchmarks.memory.throughput.stores import (
    StoreSearch,
    build_user_store,
    load_samples,
    prepare_sample_stores,
    search_store,
    search_store_for_kv,
    store_stable_id_items,
)

FACTS = ("fact-1", "fact-2")


def test_every_user_shares_one_conversation_so_requests_scale_load_over_a_fixed_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_throughput_config(tmp_path, dataset_path=tmp_path / "locomo10.json")
    loaded_paths: list[Path] = []
    monkeypatch.setattr(
        stores,
        "load_locomo",
        lambda path: (loaded_paths.append(path), ["first", "second", "third"])[1],
    )

    assert load_samples(config) == ["first"]
    assert loaded_paths == [tmp_path / "locomo10.json"]


class StoreBuilds:
    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.built: list[dict[str, Any]] = []
        monkeypatch.setattr(stores, "build_user_store", self.build)

    def build(self, config, *, backend, store_root, facts) -> str:
        self.built.append({"backend": backend, "store_root": store_root, "facts": facts})
        return f"memory of {store_root.name}"


def test_one_store_is_built_and_timed_per_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ticking_clock
) -> None:
    builds = StoreBuilds(monkeypatch)
    ticking_clock([])
    samples = [SimpleNamespace(sample_id="a"), SimpleNamespace(sample_id="b")]

    sample_stores = prepare_sample_stores(
        make_throughput_config(tmp_path),
        samples,
        backend="qdrant",
        store_root=tmp_path / "stores",
        facts_by_sample={"a": ("a-fact",), "b": ("b-fact",)},
    )

    assert builds.built == [
        {"backend": "qdrant", "store_root": tmp_path / "stores" / "a", "facts": ("a-fact",)},
        {"backend": "qdrant", "store_root": tmp_path / "stores" / "b", "facts": ("b-fact",)},
    ]
    assert sample_stores.memories == {"a": "memory of a", "b": "memory of b"}
    assert sample_stores.memory("b") == "memory of b"
    assert sample_stores.build_time_s == {"a": 3.0, "b": 7.0}
    assert sample_stores.setup_time_for_users(samples, 1) == 3.0
    assert sample_stores.setup_time_for_users(samples, 5) == 10.0


class Mem0Build:
    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.events: list[Any] = []
        self.memory = SimpleNamespace(embedding_model="live embedder")
        monkeypatch.setattr(stores, "create_mem0_memory", self.create)
        monkeypatch.setattr(stores, "CachedEmbedder", self.cached_embedder)
        monkeypatch.setattr(stores, "load_facts_into_memory", self.load_facts)
        monkeypatch.setattr(stores, "finalize_memory_stores", self.finalize)

    def record(self, step: str, *details: Any) -> None:
        self.events.append((step, self.memory.embedding_model, *details))

    def create(self, settings: Mem0Settings) -> Any:
        return self.memory

    def cached_embedder(self, wrapped: Any, **options: Any) -> str:
        self.events.append(("cache", wrapped, options))
        return "cached embedder"

    def load_facts(self, memory: Any, facts: Any, **options: Any) -> None:
        self.record("load", facts, options)

    def finalize(self, memory: Any) -> None:
        self.record("finalize")

    def build(self, tmp_path: Path, **settings: Any) -> Any:
        return build_user_store(
            make_throughput_config(tmp_path, **settings),
            backend="jasper",
            store_root=tmp_path / "store",
            facts=FACTS,
        )


@pytest.mark.parametrize("embedding_cache_enabled", [True, False])
def test_facts_are_replayed_without_entity_linking_because_no_condition_reads_the_entity_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedding_cache_enabled: bool
) -> None:
    mem0_build = Mem0Build(monkeypatch)

    mem0_build.build(tmp_path, embedding_cache_enabled=embedding_cache_enabled)

    (replay,) = [event for event in mem0_build.events if event[0] == "load"]
    assert replay[2:] == (FACTS, {"link_entities": False})


def test_fact_replay_reads_the_embedding_cache_and_queries_embed_live_again_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mem0_build = Mem0Build(monkeypatch)

    memory = mem0_build.build(tmp_path, embedding_model="embed-x")

    assert memory is mem0_build.memory
    assert mem0_build.events == [
        (
            "cache",
            "live embedder",
            {
                "cache_dir": tmp_path / "cache/embeddings",
                "model": "embed-x",
                "mode": "write",
            },
        ),
        ("load", "cached embedder", FACTS, {"link_entities": False}),
        ("finalize", "live embedder"),
    ]
    assert memory.embedding_model == "live embedder"


def memory_with_store(**store_attributes: Any) -> Any:
    return SimpleNamespace(vector_store=SimpleNamespace(**store_attributes))


def test_text_search_times_the_query_embedding_together_with_the_search(
    monkeypatch: pytest.MonkeyPatch, ticking_clock
) -> None:
    events: list[Any] = []
    ticking_clock(events)
    monkeypatch.setattr(
        stores, "embed_mem0_query", lambda memory, query: (events.append("embed"), [0.5])[1]
    )

    def search(**arguments: Any) -> Any:
        events.append(("search", arguments))
        return iter(["first", "second"])

    memory = memory_with_store(
        search=search, last_search_metrics=SimpleNamespace(search_time_ms=1.5)
    )

    found = search_store(memory, "the query", top_k=3)

    assert events == [
        "clock",
        "embed",
        ("search", {"query": "the query", "vectors": [0.5], "top_k": 3}),
        "clock",
    ]
    assert found == StoreSearch(["first", "second"], 3.0, 0.0015)


def test_device_search_times_the_query_embedding_together_with_the_search(
    monkeypatch: pytest.MonkeyPatch, ticking_clock
) -> None:
    events: list[Any] = []
    ticking_clock(events)
    monkeypatch.setattr(
        stores, "embed_mem0_query", lambda memory, query: (events.append("embed"), [0.5])[1]
    )
    device_result = object()

    def search_device(**arguments: Any) -> Any:
        events.append(("search_device", arguments))
        return device_result

    memory = memory_with_store(
        search_device=search_device, last_search_metrics=SimpleNamespace(search_time_ms=2.5)
    )

    found = search_store_for_kv(memory, "the query", top_k=3)

    assert events == [
        "clock",
        "embed",
        ("search_device", {"query": "the query", "vectors": [0.5], "top_k": 3}),
        "clock",
    ]
    assert found == StoreSearch(device_result, 3.0, 0.0025)


@pytest.mark.parametrize(
    "use_store,missing_method",
    [(store_stable_id_items, "stable_id_items")],
)
def test_device_extensions_are_checked_where_used_because_mem0_types_its_store_generically(
    use_store, missing_method: str
) -> None:
    with pytest.raises(RuntimeError) as failure:
        use_store(memory_with_store())

    assert str(failure.value) == f"Mem0 vector store has no {missing_method}()."
