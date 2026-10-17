"""Operations on a live Mem0 ``Memory`` the benchmark built: its stores, embedder, and lifetime.

These functions are the only places the benchmark reaches into Mem0's object
model. Mem0 keeps the primary vector store in ``vector_store``, its LLM in
``llm``, its embedder in ``embedding_model``, and creates the entity store on
first use through the ``entity_store`` property backed by ``_entity_store``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from benchmarks.common.embedding_cache import CachedEmbedder
from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.mem0.llm_cache import CachedMemoryLLM


class TimedEmbedder:
    """Pass-through wrapper that records embedding time when the cache is disabled."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.embed_time_ms = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def embed(self, text: Any, memory_action: str | None = None) -> Any:
        started = time.perf_counter()
        try:
            return self._wrapped.embed(text, memory_action)
        finally:
            self.embed_time_ms += (time.perf_counter() - started) * 1000

    def embed_batch(self, texts: Any, memory_action: str = "add") -> Any:
        started = time.perf_counter()
        try:
            return self._wrapped.embed_batch(texts, memory_action)
        finally:
            self.embed_time_ms += (time.perf_counter() - started) * 1000


@dataclass(slots=True)
class BuiltMemory:
    """A Mem0 memory holding one conversation's facts, with the wrappers installed on it."""

    memory: Any
    facts: tuple[MemoryFact, ...]
    metrics: dict[str, Any]
    embedder: CachedEmbedder | TimedEmbedder | None
    llm_cache: CachedMemoryLLM | None = None

    @property
    def embed_time_ms(self) -> float:
        return self.embedder.embed_time_ms if self.embedder is not None else 0.0

    def embedding_cache_stats(self) -> dict[str, Any] | None:
        if isinstance(self.embedder, CachedEmbedder):
            return self.embedder.stats()
        return None

    def llm_cache_stats(self) -> dict[str, Any] | None:
        if self.llm_cache is not None:
            return self.llm_cache.stats()
        return None

    def log_cache_stats(self, sample_id: str) -> None:
        stats = self.embedding_cache_stats()
        if stats is not None:
            logger.info(
                "Embedding cache sample_id={} mode={} hits={} misses={} dir={}",
                sample_id,
                stats["mode"],
                stats["hits"],
                stats["misses"],
                stats["cache_dir"],
            )
        llm_stats = self.llm_cache_stats()
        if llm_stats is not None:
            logger.info(
                "Mem0 inference cache sample_id={} mode={} hits={} misses={} dir={}",
                sample_id,
                llm_stats["mode"],
                llm_stats["hits"],
                llm_stats["misses"],
                llm_stats["cache_dir"],
            )

    def close(self) -> None:
        close_memory(self.memory)


def install_embedder(memory: Any, embedder: Any) -> None:
    memory.embedding_model = embedder


def memory_embedder(memory: Any) -> Any:
    return memory.embedding_model


def embed_mem0_query(memory: Any, query: str) -> Any:
    """Embed a query the way Mem0's search does, as a vector for the vector store."""
    embedder = memory.embedding_model
    if isinstance(embedder, CachedEmbedder):
        return embedder.embed_array(query, CachedEmbedder.QUERY_PURPOSE)
    return embedder.embed(query, CachedEmbedder.QUERY_PURPOSE)


def reset_memory_stores(memory: Any) -> None:
    """Empty both stores; reading ``entity_store`` creates it, which can fail on its own."""
    memory.vector_store.reset()
    try:
        entity_store = memory.entity_store
    except Exception as exc:
        raise RuntimeError("Failed to initialize the Mem0 entity store.") from exc
    entity_store.reset()


def finalize_memory_stores(memory: Any) -> None:
    memory.vector_store.finalize()
    if memory._entity_store is not None:
        memory._entity_store.finalize()


def close_memory(memory: Any) -> None:
    memory.vector_store.close()
    if memory._entity_store is not None:
        memory._entity_store.close()


def vector_store_memory_stats(memory: Any) -> dict[str, Any]:
    return dict(memory.vector_store.memory_stats())


def added_memory_texts(result: Any) -> list[str]:
    """The fact texts in a Mem0 ``add`` result."""
    return [
        text
        for text in (str(row.get("memory") or "").strip() for row in _result_rows(result))
        if text
    ]


def added_memory_ids(result: Any) -> list[str]:
    return [str(row["id"]) for row in _result_rows(result) if row.get("id") is not None]


def _result_rows(result: Any) -> list[dict[str, Any]]:
    rows = result.get("results") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]
