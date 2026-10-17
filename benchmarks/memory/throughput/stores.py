"""LoCoMo samples and the per-sample Mem0 stores the throughput conditions retrieve from."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from benchmarks.common.embedding_cache import CachedEmbedder
from benchmarks.common.vector_types import VectorStoreConfig
from benchmarks.memory.data import ConversationSample, load_locomo
from benchmarks.memory.mem0.catalog_replay import load_facts_into_memory
from benchmarks.memory.mem0.fact_catalog import FactCatalogStore, MemoryFact
from benchmarks.memory.mem0.memory_handles import (
    close_memory,
    embed_mem0_query,
    finalize_memory_stores,
    install_embedder,
    memory_embedder,
)
from benchmarks.memory.mem0.provider import Mem0Settings, create_mem0_memory
from benchmarks.memory.protocol import MEMORY_LLM_TEMPERATURE
from benchmarks.memory.throughput.config import ThroughputConfig, jasper_effective_beam_width
from benchmarks.memory.throughput.workload import samples_for_users
from inferscale.v1.types import DeviceSearchResult, SearchHit

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class StoreSearch(Generic[ResultT]):
    """One query's retrieval and its timing.

    ``result`` is the backend's answer: search hits for prompt injection, or a
    device search result whose IDs stay on the GPU for the KV lookup.
    ``elapsed_s`` covers query embedding plus search; ``search_s`` is the
    vector store's own search time.
    """

    result: ResultT
    elapsed_s: float
    search_s: float


@dataclass(slots=True)
class SampleStores:
    """One Mem0 store per sample, keyed by sample id, with how long each took to build."""

    memories: dict[str, Any] = field(default_factory=dict)
    build_time_s: dict[str, float] = field(default_factory=dict)

    def memory(self, sample_id: str) -> Any:
        return self.memories[sample_id]

    def setup_time_for_users(self, samples: Sequence[ConversationSample], num_users: int) -> float:
        """Build time of the stores the first ``num_users`` users retrieve from."""
        used = samples_for_users(samples, num_users)
        return sum(self.build_time_s[sample.sample_id] for sample in used)

    def close(self) -> None:
        for memory in self.memories.values():
            close_memory(memory)
        self.memories.clear()


def require_embedding_access(config: ThroughputConfig, condition: str) -> None:
    """Every condition embeds queries live, which needs a key or a local endpoint."""
    if not config.embedding_api_key and not config.embedding_base_url:
        raise RuntimeError(
            f"{condition} requires OPENAI_API_KEY or inferscale.embedding.base_url in the run JSON."
        )


def load_samples(config: ThroughputConfig) -> list[ConversationSample]:
    samples = load_locomo(config.dataset_path)
    if not samples:
        raise RuntimeError(f"No LoCoMo samples found in {config.dataset_path}.")
    # The throughput workload is multi-user contention over one conversation:
    # every user retrieves from the same session store, so request counts
    # scale load while the memory corpus stays fixed.
    return samples[:1]


def fact_catalog_store(config: ThroughputConfig) -> FactCatalogStore:
    return FactCatalogStore(
        config.memory_llm_cache_dir,
        provider=config.memory_llm_provider,
        model=config.memory_llm_model,
        endpoint=config.memory_llm_base_url,
        embedding_model=config.embedding_model,
        embedding_endpoint=config.embedding_base_url,
        temperature=MEMORY_LLM_TEMPERATURE,
    )


def prepare_sample_stores(
    config: ThroughputConfig,
    samples: Sequence[ConversationSample],
    *,
    backend: str,
    store_root: Path,
    facts_by_sample: dict[str, tuple[MemoryFact, ...]],
) -> SampleStores:
    """Build one store per sample, timing each; a failure closes the stores built so far."""
    stores = SampleStores()
    try:
        for sample in samples:
            started = time.perf_counter()
            stores.memories[sample.sample_id] = build_user_store(
                config,
                backend=backend,
                store_root=store_root / sample.sample_id,
                facts=facts_by_sample[sample.sample_id],
            )
            stores.build_time_s[sample.sample_id] = time.perf_counter() - started
    except BaseException:
        stores.close()
        raise
    return stores


def build_user_store(
    config: ThroughputConfig,
    *,
    backend: str,
    store_root: Path,
    facts: tuple[MemoryFact, ...],
) -> Any:
    """Replay a sample's fact catalog into a fresh store on the given vector backend."""
    memory = create_mem0_memory(
        Mem0Settings(
            store_root=store_root,
            vector_config=_vector_config(config, backend),
            embedding_model=config.embedding_model,
            embedding_api_key=config.embedding_api_key or "not-needed",
            embedding_base_url=config.embedding_base_url,
            memory_llm_provider=config.memory_llm_provider,
            memory_llm_model=config.memory_llm_model,
            memory_llm_base_url=config.memory_llm_base_url,
        )
    )
    try:
        with _cached_fact_embeddings(config, memory):
            # The conditions query the vector store directly and never read
            # the entity store, so entity linking would only be setup overhead.
            load_facts_into_memory(memory, facts, link_entities=False)
        finalize_memory_stores(memory)
    except BaseException:
        close_memory(memory)
        raise
    return memory


def search_store(memory: Any, query: str, *, top_k: int) -> StoreSearch[list[SearchHit]]:
    """Embed the query and search the store.

    Hits stay best-first; the shared prompt builder reverses them exactly once
    when it lays out the injected memory.
    """
    search = _vector_store_method(memory, "search")
    started = time.perf_counter()
    hits = list(search(query=query, vectors=embed_mem0_query(memory, query), top_k=top_k))
    return StoreSearch(hits, time.perf_counter() - started, _last_search_s(memory))


def search_store_for_kv(memory: Any, query: str, *, top_k: int) -> StoreSearch[DeviceSearchResult]:
    """Embed the query once and keep the retrieved Jasper IDs on the GPU for the KV lookup."""
    search_device = _vector_store_method(
        memory,
        "search_device",
        missing="KV retrieval requires a vector store with GPU device search.",
    )
    started = time.perf_counter()
    result = search_device(query=query, vectors=embed_mem0_query(memory, query), top_k=top_k)
    if result is None:
        raise RuntimeError(
            "KV retrieval requires GPU device results; Jasper returned no device result."
        )
    return StoreSearch(result, time.perf_counter() - started, _last_search_s(memory))


def materialize_store_device_result(memory: Any, result: DeviceSearchResult) -> list[SearchHit]:
    return list(_vector_store_method(memory, "materialize_device_result")(result))


def store_stable_id_items(memory: Any) -> tuple[tuple[int, str], ...]:
    return tuple(_vector_store_method(memory, "stable_id_items")())


def _vector_store_method(memory: Any, name: str, *, missing: str | None = None) -> Callable:
    # Mem0 types ``vector_store`` as its generic VectorStoreBase, and the device
    # search, result materialization, and stable-id extensions exist only on
    # the benchmark's Jasper adapter, so each capability is checked where it
    # is used instead of assumed from the type.
    method = getattr(memory.vector_store, name, None)
    if not callable(method):
        raise RuntimeError(missing or f"Mem0 vector store has no {name}().")
    return method


def _last_search_s(memory: Any) -> float:
    return float(memory.vector_store.last_search_metrics.search_time_ms) / 1000


@contextmanager
def _cached_fact_embeddings(config: ThroughputConfig, memory: Any) -> Iterator[None]:
    """Serve fact embeddings from the shared cache while the catalog is replayed.

    The cache is free and offline after the preembed stage. The live embedder
    is restored afterwards so query-time retrieval measures live embedding
    latency rather than cache reads.
    """
    live = memory_embedder(memory)
    if not config.embedding_cache_enabled:
        yield
        return
    cached = CachedEmbedder(
        live,
        cache_dir=config.embedding_cache_dir,
        model=config.embedding_model,
        mode="write",
        endpoint=config.embedding_base_url,
    )
    install_embedder(memory, cached)
    try:
        yield
    finally:
        install_embedder(memory, live)


def _vector_config(config: ThroughputConfig, backend: str) -> VectorStoreConfig:
    beam_width = config.jasper_beam_width
    if backend == "jasper":
        beam_width = jasper_effective_beam_width(config)
    return VectorStoreConfig(
        backend=backend,
        n_neighbors=config.jasper_n_neighbors,
        alpha=config.jasper_alpha,
        workspace_budget=config.jasper_workspace_budget,
        beam_width=beam_width,
    )
