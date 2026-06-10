from __future__ import annotations

from typing import Any

from loguru import logger

from .config import BenchmarkConfig
from .data import ConversationSample, format_turn_for_memory
from .embedding.cache import CacheMode, CachedEmbedder
from .mem0_provider import create_mem0_memory
from .vector_types import VectorStoreConfig


class SampleMemoryBuilder:
    def __init__(self, config: BenchmarkConfig, *, embedding_cache_mode: CacheMode = "read") -> None:
        self.config = config
        self.embedding_cache_mode = embedding_cache_mode

    def build(self, sample: ConversationSample, *, finalize_index: bool = True) -> Any:
        store_root = self.config.run_dir / "mem0" / sample.sample_id
        memory = create_mem0_memory(
            store_root=store_root,
            vector_config=_store_config(self.config),
            embedding_model=self.config.embedding_model,
            embedding_api_key=self.config.embedding_api_key,
            embedding_base_url=self.config.embedding_base_url,
        )
        self._install_embedding_cache(memory)

        for turn in sample.turns:
            text = format_turn_for_memory(turn)
            metadata = {
                "user_id": sample.sample_id,
                "sample_id": sample.sample_id,
                "turn_id": turn.id,
                "dia_id": turn.dia_id,
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

        if finalize_index:
            logger.info("Building {} index for sample_id={}", self.config.vector_backend, sample.sample_id)
            self._finalize(memory)
            logger.info("Index ready sample_id={} backend={}", sample.sample_id, self.config.vector_backend)
        return memory

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
    return memory_embedder(memory).embed(query, "search")


def _store_config(config: BenchmarkConfig) -> VectorStoreConfig:
    return VectorStoreConfig(
        backend=config.vector_backend,
        distance=config.vector_distance,
        normalize_vectors=config.vector_normalize,
        n_neighbors=config.jasper_n_neighbors,
        alpha=config.jasper_alpha,
        workspace_budget=config.jasper_workspace_budget,
        beam_width=config.jasper_beam_width,
    )
