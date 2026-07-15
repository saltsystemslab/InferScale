"""Sample loading and per-sample mem0 vector stores for the throughput conditions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..data import ConversationSample, load_locomo
from ..embedding.cache import CachedEmbedder
from ..retrieval.fact_catalog import FactCatalogStore, MemoryFact
from ..retrieval.mem0_provider import MEMORY_LLM_TEMPERATURE, create_mem0_memory
from ..retrieval.memory_builder import embed_mem0_query, load_facts_into_memory
from ..vector_types import VectorStoreConfig
from .config import ThroughputConfig


def load_samples(config: ThroughputConfig) -> list[ConversationSample]:
    samples = load_locomo(config.dataset_path)
    if not samples:
        raise RuntimeError(f"No LoCoMo samples found in {config.dataset_path}.")
    # The throughput workload is multi-user contention over ONE conversation:
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


def build_user_store(
    config: ThroughputConfig,
    *,
    backend: str,
    store_root: Path,
    facts: tuple[MemoryFact, ...],
) -> Any:
    """Replay a sample's fact catalog into a fresh per-user store.

    Fact embeddings go through the shared cache (free and offline after
    --preembed-only), but the raw embedder is restored before returning so
    query-time retrieval measures live embedding latency, not cache reads.
    """
    memory = create_mem0_memory(
        store_root=store_root,
        vector_config=_vector_config(config, backend),
        embedding_model=config.embedding_model,
        embedding_api_key=config.embedding_api_key or "not-needed",
        embedding_base_url=config.embedding_base_url,
        memory_llm_provider=config.memory_llm_provider,
        memory_llm_model=config.memory_llm_model,
        memory_llm_base_url=config.memory_llm_base_url,
    )
    raw_embedder = getattr(memory, "embedding_model", None) or getattr(memory, "embedder", None)
    if config.embedding_cache_enabled and raw_embedder is not None:
        cached = CachedEmbedder(
            raw_embedder,
            cache_dir=config.embedding_cache_dir,
            model=config.embedding_model,
            mode="write",
            endpoint=config.embedding_base_url,
        )
        if hasattr(memory, "embedding_model"):
            memory.embedding_model = cached
        if hasattr(memory, "embedder"):
            memory.embedder = cached
    try:
        # search_store queries the vector store directly and never reads the
        # entity store, so entity linking would be pure setup overhead here.
        load_facts_into_memory(memory, facts, link_entities=False)
        _finalize_mem0(memory)
    except BaseException:
        close_mem0(memory)
        raise
    if raw_embedder is not None:
        if hasattr(memory, "embedding_model"):
            memory.embedding_model = raw_embedder
        if hasattr(memory, "embedder"):
            memory.embedder = raw_embedder
    return memory


def search_store(memory: Any, query: str, *, top_k: int) -> tuple[list[Any], float, float]:
    """Embed the query and search the store; returns (hits, elapsed_s, backend_search_s).

    Hits remain best-first; the shared accuracy prompt builder reverses them
    exactly once when it constructs the injected memory sequence.
    """
    retrieval_started = time.perf_counter()
    query_embedding = embed_mem0_query(memory, query)
    vector_store = getattr(memory, "vector_store", None)
    search = getattr(vector_store, "search", None)
    if not callable(search):
        raise RuntimeError("Mem0 memory has no searchable vector_store.")
    hits = list(
        search(
            query=query,
            vectors=query_embedding,
            top_k=top_k,
        )
    )
    elapsed_s = time.perf_counter() - retrieval_started
    metrics = getattr(vector_store, "last_search_metrics", None)
    search_s = float(getattr(metrics, "search_time_ms", 0.0) or 0.0) / 1000
    return hits, elapsed_s, search_s


def _vector_config(config: ThroughputConfig, backend: str) -> VectorStoreConfig:
    beam_width = config.jasper_beam_width
    if backend == "jasper":
        beam_width = max(beam_width, config.top_k)
    return VectorStoreConfig(
        backend=backend,
        n_neighbors=config.jasper_n_neighbors,
        alpha=config.jasper_alpha,
        workspace_budget=config.jasper_workspace_budget,
        beam_width=beam_width,
    )


def _finalize_mem0(memory: Any) -> None:
    vector_store = getattr(memory, "vector_store", None)
    finalize = getattr(vector_store, "finalize", None)
    if callable(finalize):
        finalize()


def close_mem0(memory: Any) -> None:
    vector_store = getattr(memory, "vector_store", None)
    close = getattr(vector_store, "close", None)
    if callable(close):
        close()
