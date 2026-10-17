"""Build one conversation's Mem0 memory: extract a fact catalog, or replay a saved one.

The preembed stage runs the builder in write mode, which calls the extraction
LLM and writes the catalog. Every timed stage runs it in read mode, which
replays the catalog so the LLM is never on the measured path.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.embedding_cache import CachedEmbedder, CacheMode
from benchmarks.common.vector_types import VectorStoreConfig
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, load_locomo
from benchmarks.memory.mem0.catalog_extraction import extract_facts
from benchmarks.memory.mem0.catalog_replay import load_facts_into_memory
from benchmarks.memory.mem0.fact_catalog import FactCatalogStore, MemoryFact
from benchmarks.memory.mem0.llm_cache import CachedMemoryLLM
from benchmarks.memory.mem0.memory_handles import (
    BuiltMemory,
    TimedEmbedder,
    close_memory,
    finalize_memory_stores,
    install_embedder,
    reset_memory_stores,
    vector_store_memory_stats,
)
from benchmarks.memory.mem0.prepared_retriever import PreparedMem0Retriever
from benchmarks.memory.mem0.provider import Mem0Settings, create_mem0_memory, resolved_mem0_backend
from benchmarks.memory.protocol import MEMORY_LLM_TEMPERATURE


def fact_catalog_store_for(config: MemoryRunConfig) -> FactCatalogStore:
    return FactCatalogStore(
        config.memory_llm_cache_dir,
        provider=config.memory_llm_provider,
        model=config.memory_llm_model,
        endpoint=config.memory_llm_base_url,
        embedding_model=config.embedding_model,
        embedding_endpoint=config.embedding_base_url,
        temperature=MEMORY_LLM_TEMPERATURE,
    )


def missing_fact_catalogs(config: MemoryRunConfig) -> list[tuple[str, Path]]:
    """Sample ids and catalog paths that do not exist for this config's catalog identity."""
    store = fact_catalog_store_for(config)
    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    return [
        (sample.sample_id, store.path_for(sample))
        for sample in samples
        if not store.path_for(sample).exists()
    ]


def vector_store_config(
    config: MemoryRunConfig, *, backend: str | None = None
) -> VectorStoreConfig:
    return VectorStoreConfig(
        backend=backend or config.vector_backend,
        n_neighbors=config.jasper_n_neighbors,
        alpha=config.jasper_alpha,
        workspace_budget=config.jasper_workspace_budget,
        beam_width=config.jasper_effective_beam_width or config.jasper_beam_width,
    )


class SampleMemoryBuilder:
    def __init__(
        self,
        config: MemoryRunConfig,
        *,
        embedding_cache_mode: CacheMode = "read",
        memory_llm_cache_mode: CacheMode | None = None,
    ) -> None:
        self.config = config
        self.embedding_cache_mode = embedding_cache_mode
        self.memory_llm_cache_mode = memory_llm_cache_mode or embedding_cache_mode
        self._fact_catalog_store = fact_catalog_store_for(config)

    def load_fact_catalog(self, sample: ConversationSample) -> tuple[MemoryFact, ...]:
        return self._fact_catalog_store.load(sample)

    def build_memory(
        self, sample: ConversationSample, *, finalize_index: bool = True
    ) -> BuiltMemory:
        """Extract the catalog in write mode; replay the saved catalog otherwise."""
        if self.memory_llm_cache_mode == "write":
            return self._extract(sample)
        return self._replay(sample, finalize_index=finalize_index)

    def build_retriever_with_metrics(
        self, sample: ConversationSample, *, finalize_index: bool = True
    ) -> tuple[PreparedMem0Retriever, dict[str, Any]]:
        if self.memory_llm_cache_mode == "write":
            raise RuntimeError(
                "Prepared retrievers consume immutable fact catalogs. Run build_memory() "
                "during the preembed stage first, then construct a read-mode SampleMemoryBuilder."
            )
        built = self.build_memory(sample, finalize_index=finalize_index)
        retriever = PreparedMem0Retriever(
            built, sample_id=sample.sample_id, vector_backend=resolved_mem0_backend(built.memory)
        )
        return retriever, built.metrics

    def _replay(self, sample: ConversationSample, *, finalize_index: bool) -> BuiltMemory:
        facts = self.load_fact_catalog(sample)
        total_started = time.perf_counter()
        store_root = self.config.local_store_scratch_dir() / "mem0" / sample.sample_id
        memory, embedder = self._create_memory(store_root, vector_store_config(self.config))
        try:
            backend = resolved_mem0_backend(memory)
            create_ms = (time.perf_counter() - total_started) * 1000
            build_started = time.perf_counter()
            load_facts_into_memory(memory, facts)
            logger.info(
                "Loaded {} immutable Mem0 facts for sample_id={} backend={}",
                len(facts),
                sample.sample_id,
                backend,
            )
            build_ms = (time.perf_counter() - build_started) * 1000
            index_ms = self._finalize(memory, sample, backend) if finalize_index else None
            metrics = {
                "vector_backend": backend,
                "memory_create_time_ms": create_ms,
                "embedding_memory_build_time_ms": build_ms,
                "vector_index_build_time_ms": index_ms,
                "memory_setup_time_ms": (time.perf_counter() - total_started) * 1000,
                "memory_input_turn_count": len(sample.turns),
                "memory_inferred_record_count": len(facts),
                "memory_fact_catalog_loaded": 1,
                **vector_store_memory_stats(memory),
            }
            return BuiltMemory(memory=memory, facts=facts, metrics=metrics, embedder=embedder)
        except BaseException:
            close_memory(memory)
            raise

    def _extract(self, sample: ConversationSample) -> BuiltMemory:
        total_started = time.perf_counter()
        store_root = self.config.local_store_scratch_dir() / "mem0-extraction" / sample.sample_id
        # Mem0 keeps a messages table (its last ten messages feed the extraction
        # prompt) in history.sqlite under the store root, so a repeated
        # extraction must start from an empty directory.
        shutil.rmtree(store_root, ignore_errors=True)
        memory, embedder = self._create_memory(
            store_root, vector_store_config(self.config, backend="qdrant")
        )
        try:
            llm_cache = self._install_llm_cache(memory)
            create_ms = (time.perf_counter() - total_started) * 1000
            build_started = time.perf_counter()
            facts = tuple(extract_facts(memory, sample, embedder=embedder))
            catalog_path = self._fact_catalog_store.write(sample, facts)
            logger.info(
                "Materialized {} inferred Mem0 facts for sample_id={} catalog={}",
                len(facts),
                sample.sample_id,
                catalog_path,
            )
            metrics = {
                "vector_backend": "qdrant",
                "memory_create_time_ms": create_ms,
                "embedding_memory_build_time_ms": (time.perf_counter() - build_started) * 1000,
                "vector_index_build_time_ms": None,
                "memory_setup_time_ms": (time.perf_counter() - total_started) * 1000,
                "memory_input_turn_count": len(sample.turns),
                "memory_inferred_record_count": len(facts),
                "memory_fact_catalog_loaded": 0,
                "memory_llm_cache_hits": llm_cache.hits,
                "memory_llm_cache_misses": llm_cache.misses,
            }
            return BuiltMemory(
                memory=memory, facts=facts, metrics=metrics, embedder=embedder, llm_cache=llm_cache
            )
        except BaseException:
            close_memory(memory)
            raise

    def _create_memory(
        self, store_root: Path, vector_config: VectorStoreConfig
    ) -> tuple[Any, CachedEmbedder | TimedEmbedder]:
        """A fresh memory with the embedder wrapper installed and both stores emptied."""
        memory = create_mem0_memory(
            Mem0Settings(
                store_root=store_root,
                vector_config=vector_config,
                embedding_model=self.config.embedding_model,
                embedding_api_key=self.config.embedding_api_key,
                embedding_base_url=self.config.embedding_base_url,
                memory_llm_provider=self.config.memory_llm_provider,
                memory_llm_model=self.config.memory_llm_model,
                memory_llm_api_key=self.config.memory_llm_api_key,
                memory_llm_base_url=self.config.memory_llm_base_url,
            )
        )
        embedder = self._embedder_wrapper(memory.embedding_model)
        install_embedder(memory, embedder)
        reset_memory_stores(memory)
        return memory, embedder

    def _embedder_wrapper(self, raw_embedder: Any) -> CachedEmbedder | TimedEmbedder:
        if not self.config.embedding_cache_enabled:
            logger.info("Embedding cache disabled")
            return TimedEmbedder(raw_embedder)
        cached = CachedEmbedder(
            raw_embedder,
            cache_dir=self.config.embedding_cache_dir,
            model=self.config.embedding_model,
            mode=self.embedding_cache_mode,
            endpoint=self.config.embedding_base_url,
        )
        logger.info(
            "Embedding cache enabled mode={} dir={}", self.embedding_cache_mode, cached.cache_dir
        )
        return cached

    def _install_llm_cache(self, memory: Any) -> CachedMemoryLLM:
        cached = CachedMemoryLLM(
            memory.llm,
            cache_dir=self.config.memory_llm_cache_dir,
            provider=self.config.memory_llm_provider,
            model=self.config.memory_llm_model,
            mode=self.memory_llm_cache_mode,
            endpoint=self.config.memory_llm_base_url,
            temperature=MEMORY_LLM_TEMPERATURE,
        )
        memory.llm = cached
        logger.info(
            "Mem0 inference cache enabled mode={} provider={} model={} dir={}",
            self.memory_llm_cache_mode,
            self.config.memory_llm_provider,
            self.config.memory_llm_model,
            cached.cache_dir,
        )
        return cached

    @staticmethod
    def _finalize(memory: Any, sample: ConversationSample, backend: str) -> float:
        logger.info("Building vector index for sample_id={} backend={}", sample.sample_id, backend)
        started = time.perf_counter()
        finalize_memory_stores(memory)
        logger.info("Index ready sample_id={} backend={}", sample.sample_id, backend)
        return (time.perf_counter() - started) * 1000
