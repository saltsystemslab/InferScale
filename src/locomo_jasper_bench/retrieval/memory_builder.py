from __future__ import annotations

import time
from typing import Any

from loguru import logger

from ..config import BenchmarkConfig
from ..data import ConversationSample, format_turn_for_memory
from ..embedding.cache import CacheMode, CachedEmbedder
from ..vector_types import VectorStoreConfig
from .mem0_provider import create_mem0_memory


class SampleMemoryBuilder:
    def __init__(self, config: BenchmarkConfig, *, embedding_cache_mode: CacheMode = "read") -> None:
        self.config = config
        self.embedding_cache_mode = embedding_cache_mode

    def build(self, sample: ConversationSample, *, finalize_index: bool = True) -> Any:
        memory, _ = self.build_with_metrics(sample, finalize_index=finalize_index)
        return memory

    def build_with_metrics(self, sample: ConversationSample, *, finalize_index: bool = True) -> tuple[Any, dict[str, Any]]:
        store_root = self.config.run_dir / "mem0" / sample.sample_id
        total_started = time.perf_counter()
        create_started = time.perf_counter()
        memory = create_mem0_memory(
            store_root=store_root,
            vector_config=_store_config(self.config),
            embedding_model=self.config.embedding_model,
            embedding_api_key=self.config.embedding_api_key,
            embedding_base_url=self.config.embedding_base_url,
        )
        self._install_embedding_cache(memory)
        memory_create_time_ms = (time.perf_counter() - create_started) * 1000

        build_started = time.perf_counter()
        for turn in sample.turns:
            text = format_turn_for_memory(turn)
            metadata = {
                "user_id": sample.sample_id,
                "sample_id": sample.sample_id,
                "turn_id": turn.id,
                "session_id": turn.session_id,
                "turn_index": turn.turn_index,
                "speaker": turn.speaker,
                "timestamp": turn.timestamp,
            }
            memory.add(
                [{"role": "user", "content": text}],
                user_id=sample.sample_id,
                infer=False,
                metadata=metadata,
            )
        logger.info(
            "Added {} LoCoMo turns to Mem0 for sample_id={} infer=false",
            len(sample.turns),
            sample.sample_id,
        )
        embedding_memory_build_time_ms = (time.perf_counter() - build_started) * 1000

        vector_index_build_time_ms = None
        if finalize_index:
            logger.info("Building vector index for sample_id={} backend={}", sample.sample_id, self.config.vector_backend)
            index_started = time.perf_counter()
            self._finalize(memory)
            vector_index_build_time_ms = (time.perf_counter() - index_started) * 1000
            logger.info("Index ready sample_id={} backend={}", sample.sample_id, self.config.vector_backend)
        metrics = {
            "memory_create_time_ms": memory_create_time_ms,
            "embedding_memory_build_time_ms": embedding_memory_build_time_ms,
            "vector_index_build_time_ms": vector_index_build_time_ms,
            "memory_setup_time_ms": (time.perf_counter() - total_started) * 1000,
        }
        return memory, metrics

    def log_embedding_cache_stats(self, memory: Any, sample_id: str) -> None:
        stats = self.embedding_cache_stats(memory)
        if stats is None:
            return
        logger.info(
            "Embedding cache sample_id={} mode={} hits={} misses={} dir={}",
            sample_id,
            stats["mode"],
            stats["hits"],
            stats["misses"],
            stats["cache_dir"],
        )

    def embedding_cache_stats(self, memory: Any) -> dict[str, Any] | None:
        cache = getattr(memory, "_locomo_embedding_cache", None)
        if not isinstance(cache, CachedEmbedder):
            return None
        return cache.stats()

    def close(self, memory: Any) -> None:
        vector_store = getattr(memory, "vector_store", None)
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()

    def _install_embedding_cache(self, memory: Any) -> None:
        if not self.config.embedding_cache_enabled:
            logger.info("Embedding cache disabled")
            return

        embedder = getattr(memory, "embedding_model", None) or getattr(memory, "embedder", None)
        if embedder is None:
            logger.warning("Embedding cache requested but Mem0 memory has no embedder attribute")
            return

        cached = CachedEmbedder(
            embedder,
            cache_dir=self.config.embedding_cache_dir,
            model=self.config.embedding_model,
            mode=self.embedding_cache_mode,
        )
        if hasattr(memory, "embedding_model"):
            memory.embedding_model = cached
        if hasattr(memory, "embedder"):
            memory.embedder = cached
        memory._locomo_embedding_cache = cached
        logger.info("Embedding cache enabled mode={} dir={}", self.embedding_cache_mode, cached.cache_dir)

    def _finalize(self, memory: Any) -> None:
        vector_store = getattr(memory, "vector_store", None)
        finalize = getattr(vector_store, "finalize", None)
        if callable(finalize):
            finalize()


def memory_embedder(memory: Any) -> Any:
    embedder = getattr(memory, "embedding_model", None) or getattr(memory, "embedder", None)
    embed = getattr(embedder, "embed", None)
    if not callable(embed):
        raise RuntimeError("Mem0 memory has no callable embedder.")
    return embedder


def embed_mem0_query(memory: Any, query: str) -> Any:
    embedder = memory_embedder(memory)
    embed_array = getattr(embedder, "embed_array", None)
    if callable(embed_array):
        return embed_array(query, "search")
    return embedder.embed(query, "search")


def _store_config(config: BenchmarkConfig) -> VectorStoreConfig:
    return VectorStoreConfig(
        backend=config.vector_backend,
        distance=config.vector_distance,
        n_neighbors=config.jasper_n_neighbors,
        alpha=config.jasper_alpha,
        workspace_budget=config.jasper_workspace_budget,
        beam_width=config.jasper_beam_width,
    )
