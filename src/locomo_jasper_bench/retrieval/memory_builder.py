from __future__ import annotations

import importlib
import os
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from loguru import logger

from pathlib import Path

from ..config import BenchmarkConfig
from ..data import ConversationSample, format_turn_for_memory, load_locomo
from ..embedding.cache import CacheMode, CachedEmbedder
from ..runtime_paths import default_mem0_dir_string, local_store_scratch_dir
from ..vector_types import VectorStoreConfig
from .fact_catalog import FactCatalogStore, MemoryFact, make_memory_fact, source_metadata
from .mem0_provider import MEMORY_LLM_TEMPERATURE, create_mem0_memory, resolved_mem0_backend
from .memory_llm_cache import CachedMemoryLLM
from .prepared_retriever import PreparedMem0Retriever


_MEM0_PROMPT_PATCH_LOCK = threading.Lock()
_MEM0_OBSERVATION_DATE: ContextVar[str | None] = ContextVar(
    "mem0_observation_date",
    default=None,
)


def fact_catalog_store_for(config: BenchmarkConfig) -> FactCatalogStore:
    return FactCatalogStore(
        config.memory_llm_cache_dir,
        provider=config.memory_llm_provider,
        model=config.memory_llm_model,
        endpoint=config.memory_llm_base_url,
        embedding_model=config.embedding_model,
        embedding_endpoint=config.embedding_base_url,
        temperature=MEMORY_LLM_TEMPERATURE,
    )


def missing_fact_catalogs(config: BenchmarkConfig) -> list[tuple[str, Path]]:
    """Sample ids and expected catalog paths that do not exist for this config's full catalog identity."""
    store = fact_catalog_store_for(config)
    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    return [
        (sample.sample_id, store.path_for(sample))
        for sample in samples
        if not store.path_for(sample).exists()
    ]


class SampleMemoryBuilder:
    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        embedding_cache_mode: CacheMode = "read",
        memory_llm_cache_mode: CacheMode | None = None,
    ) -> None:
        self.config = config
        self.embedding_cache_mode = embedding_cache_mode
        self.memory_llm_cache_mode = memory_llm_cache_mode or embedding_cache_mode
        self._fact_catalog_store = fact_catalog_store_for(config)

    def build(self, sample: ConversationSample, *, finalize_index: bool = True) -> Any:
        memory, _ = self.build_with_metrics(sample, finalize_index=finalize_index)
        return memory

    def build_retriever(
        self,
        sample: ConversationSample,
        *,
        finalize_index: bool = True,
    ) -> PreparedMem0Retriever:
        retriever, _ = self.build_retriever_with_metrics(
            sample,
            finalize_index=finalize_index,
        )
        return retriever

    def build_retriever_with_metrics(
        self,
        sample: ConversationSample,
        *,
        finalize_index: bool = True,
    ) -> tuple[PreparedMem0Retriever, dict[str, Any]]:
        if self.memory_llm_cache_mode == "write":
            raise RuntimeError(
                "Prepared retrievers consume immutable fact catalogs. Run build_with_metrics() "
                "during --preembed-only first, then construct a read-mode SampleMemoryBuilder."
            )
        memory, metrics = self.build_with_metrics(sample, finalize_index=finalize_index)
        facts = self.load_fact_catalog(sample)
        return (
            PreparedMem0Retriever(
                memory,
                sample_id=sample.sample_id,
                fact_catalog=facts,
                vector_backend=resolved_mem0_backend(memory),
            ),
            metrics,
        )

    def load_fact_catalog(self, sample: ConversationSample) -> tuple[MemoryFact, ...]:
        return self._fact_catalog_store.load(sample)

    def build_with_metrics(
        self,
        sample: ConversationSample,
        *,
        finalize_index: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        if self.memory_llm_cache_mode == "write":
            return self._extract_and_materialize_catalog(sample)

        facts = self.load_fact_catalog(sample)
        store_root = local_store_scratch_dir(self.config.run_id) / "mem0" / sample.sample_id
        total_started = time.perf_counter()
        create_started = time.perf_counter()
        memory = self._create_memory(
            store_root,
            _store_config(self.config),
            inference_enabled=False,
        )
        resolved_backend = resolved_mem0_backend(memory)
        self._install_embedding_cache(memory)
        self._reset_vector_store(memory)
        memory_create_time_ms = (time.perf_counter() - create_started) * 1000

        build_started = time.perf_counter()
        self._load_facts_into_memory(memory, facts)
        logger.info(
            "Loaded {} immutable Mem0 facts for sample_id={} backend={}",
            len(facts),
            sample.sample_id,
            resolved_backend,
        )
        embedding_memory_build_time_ms = (time.perf_counter() - build_started) * 1000

        vector_index_build_time_ms = None
        if finalize_index:
            logger.info("Building vector index for sample_id={} backend={}", sample.sample_id, resolved_backend)
            index_started = time.perf_counter()
            self._finalize(memory)
            vector_index_build_time_ms = (time.perf_counter() - index_started) * 1000
            logger.info("Index ready sample_id={} backend={}", sample.sample_id, resolved_backend)
        metrics = {
            "vector_backend": resolved_backend,
            "memory_create_time_ms": memory_create_time_ms,
            "embedding_memory_build_time_ms": embedding_memory_build_time_ms,
            "vector_index_build_time_ms": vector_index_build_time_ms,
            "memory_setup_time_ms": (time.perf_counter() - total_started) * 1000,
            "memory_input_turn_count": len(sample.turns),
            "memory_inferred_record_count": len(facts),
            "memory_fact_catalog_loaded": 1,
        }
        metrics.update(self._vector_store_memory_stats(memory))
        return memory, metrics

    def _extract_and_materialize_catalog(
        self,
        sample: ConversationSample,
    ) -> tuple[Any, dict[str, Any]]:
        total_started = time.perf_counter()
        create_started = time.perf_counter()
        store_root = local_store_scratch_dir(self.config.run_id) / "mem0-extraction" / sample.sample_id
        # Mem0 persists a messages table (last 10 messages feed the extraction
        # prompt) in history.sqlite under the store root; wipe the whole staging
        # directory so a repeated extraction run starts clean.
        shutil.rmtree(store_root, ignore_errors=True)
        staging_config = _store_config(self.config, backend="qdrant")
        memory = self._create_memory(
            store_root,
            staging_config,
            inference_enabled=True,
        )
        try:
            self._install_embedding_cache(memory)
            self._install_memory_llm_cache(memory)
            self._reset_vector_store(memory)
            memory_create_time_ms = (time.perf_counter() - create_started) * 1000

            build_started = time.perf_counter()
            facts: list[MemoryFact] = []
            seen_fact_ids: set[str] = set()
            for turn in sample.turns:
                metadata = source_metadata(sample, turn)
                role = str(metadata["role"])
                with _mem0_observation_date(str(metadata["created_at"])):
                    result = memory.add(
                        [{"role": role, "content": format_turn_for_memory(turn)}],
                        user_id=sample.sample_id,
                        infer=True,
                        metadata=metadata,
                    )
                for text in _memory_result_texts(result):
                    fact = make_memory_fact(text, sample, turn)
                    if fact.id not in seen_fact_ids:
                        facts.append(fact)
                        seen_fact_ids.add(fact.id)
            # Mem0's merge phase can store a final fact text that differs
            # from the string it embedded (e.g. an update rewrites the text
            # after the add-path embedding), so a catalog replay would miss
            # the read-mode cache. Write-through-embed every final text so
            # catalogs are replay-complete by construction; cache hits make
            # this free for the common case.
            embedder = memory_embedder(memory)
            for fact in facts:
                embedder.embed(fact.text, "add")
            catalog_path = self._fact_catalog_store.write(sample, facts)
            memory._locomo_fact_catalog = tuple(facts)
            logger.info(
                "Materialized {} inferred Mem0 facts for sample_id={} catalog={}",
                len(facts),
                sample.sample_id,
                catalog_path,
            )
            build_time_ms = (time.perf_counter() - build_started) * 1000
            cache_stats = self.memory_llm_cache_stats(memory)
            metrics: dict[str, Any] = {
                "vector_backend": "qdrant",
                "memory_create_time_ms": memory_create_time_ms,
                "embedding_memory_build_time_ms": build_time_ms,
                "vector_index_build_time_ms": None,
                "memory_setup_time_ms": (time.perf_counter() - total_started) * 1000,
                "memory_input_turn_count": len(sample.turns),
                "memory_inferred_record_count": len(facts),
                "memory_fact_catalog_loaded": 0,
            }
            if cache_stats is not None:
                metrics["memory_llm_cache_hits"] = int(cache_stats["hits"])
                metrics["memory_llm_cache_misses"] = int(cache_stats["misses"])
            return memory, metrics
        except BaseException:
            self.close(memory)
            raise

    def _create_memory(
        self,
        store_root: Any,
        vector_config: VectorStoreConfig,
        *,
        inference_enabled: bool,
    ) -> Any:
        del inference_enabled  # The vllm memory LLM never needs credentials to construct.
        return create_mem0_memory(
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

    def _load_facts_into_memory(
        self,
        memory: Any,
        facts: tuple[MemoryFact, ...],
    ) -> None:
        load_facts_into_memory(memory, facts)

    @staticmethod
    def _reset_vector_store(memory: Any) -> None:
        stores = [getattr(memory, "vector_store", None)]
        try:
            stores.append(getattr(memory, "entity_store", None))
        except Exception as exc:
            raise RuntimeError("Failed to initialize the Mem0 entity store.") from exc
        for vector_store in stores:
            reset = getattr(vector_store, "reset", None)
            if callable(reset):
                reset()

    def log_embedding_cache_stats(self, memory: Any, sample_id: str) -> None:
        stats = self.embedding_cache_stats(memory)
        if stats is not None:
            logger.info(
                "Embedding cache sample_id={} mode={} hits={} misses={} dir={}",
                sample_id,
                stats["mode"],
                stats["hits"],
                stats["misses"],
                stats["cache_dir"],
            )
        memory_llm_stats = self.memory_llm_cache_stats(memory)
        if memory_llm_stats is not None:
            logger.info(
                "Mem0 inference cache sample_id={} mode={} hits={} misses={} dir={}",
                sample_id,
                memory_llm_stats["mode"],
                memory_llm_stats["hits"],
                memory_llm_stats["misses"],
                memory_llm_stats["cache_dir"],
            )

    def embedding_cache_stats(self, memory: Any) -> dict[str, Any] | None:
        cache = getattr(memory, "_locomo_embedding_cache", None)
        if not isinstance(cache, CachedEmbedder):
            return None
        return cache.stats()

    def memory_llm_cache_stats(self, memory: Any) -> dict[str, Any] | None:
        cache = getattr(memory, "_locomo_memory_llm_cache", None)
        if not isinstance(cache, CachedMemoryLLM):
            return None
        return cache.stats()

    def close(self, memory: Any) -> None:
        for vector_store in (
            getattr(memory, "vector_store", None),
            getattr(memory, "_entity_store", None),
        ):
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
            endpoint=self.config.embedding_base_url,
        )
        if hasattr(memory, "embedding_model"):
            memory.embedding_model = cached
        if hasattr(memory, "embedder"):
            memory.embedder = cached
        memory._locomo_embedding_cache = cached
        logger.info("Embedding cache enabled mode={} dir={}", self.embedding_cache_mode, cached.cache_dir)

    def _install_memory_llm_cache(self, memory: Any) -> None:
        llm = getattr(memory, "llm", None)
        if llm is None:
            raise RuntimeError("Mem0 inference requires a callable LLM client, but Memory.llm is unavailable.")
        generate_response = getattr(llm, "generate_response", None)
        if not callable(generate_response):
            raise RuntimeError("Mem0 inference requires Memory.llm.generate_response().")

        cached = CachedMemoryLLM(
            llm,
            cache_dir=self.config.memory_llm_cache_dir,
            provider=self.config.memory_llm_provider,
            model=self.config.memory_llm_model,
            mode=self.memory_llm_cache_mode,
            endpoint=self.config.memory_llm_base_url,
            temperature=MEMORY_LLM_TEMPERATURE,
        )
        memory.llm = cached
        memory._locomo_memory_llm_cache = cached
        logger.info(
            "Mem0 inference cache enabled mode={} provider={} model={} dir={}",
            self.memory_llm_cache_mode,
            self.config.memory_llm_provider,
            self.config.memory_llm_model,
            cached.cache_dir,
        )

    def _finalize(self, memory: Any) -> None:
        for vector_store in (
            getattr(memory, "vector_store", None),
            getattr(memory, "_entity_store", None),
        ):
            finalize = getattr(vector_store, "finalize", None)
            if callable(finalize):
                finalize()

    def _vector_store_memory_stats(self, memory: Any) -> dict[str, Any]:
        vector_store = getattr(memory, "vector_store", None)
        memory_stats = getattr(vector_store, "memory_stats", None)
        if callable(memory_stats):
            return dict(memory_stats())
        return {}


def load_facts_into_memory(
    memory: Any,
    facts: tuple[MemoryFact, ...],
    *,
    link_entities: bool = True,
) -> None:
    """Replay immutable catalog facts into a live Mem0 store with infer=False.

    link_entities=False skips populating the entity store; callers that search
    the vector store directly (the throughput worker) never read it.
    """
    link_entity = getattr(memory, "_link_entities_for_memory", None) if link_entities else None
    for fact in facts:
        result = memory.add(
            [{"role": "user", "content": fact.text}],
            user_id=fact.sample_id,
            infer=False,
            metadata=fact.metadata(),
        )
        if callable(link_entity) and _memory_result_ids(result):
            link_entity(
                fact.id,
                fact.text,
                {"user_id": fact.sample_id},
            )
    memory._locomo_fact_catalog = facts


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


def _store_config(
    config: BenchmarkConfig,
    *,
    backend: str | None = None,
) -> VectorStoreConfig:
    return VectorStoreConfig(
        backend=backend or config.vector_backend,
        n_neighbors=config.jasper_n_neighbors,
        alpha=config.jasper_alpha,
        workspace_budget=config.jasper_workspace_budget,
        beam_width=config.jasper_effective_beam_width or config.jasper_beam_width,
    )


def _memory_result_texts(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    memories = result.get("results")
    if not isinstance(memories, list):
        return []
    return [
        str(memory.get("memory") or "").strip()
        for memory in memories
        if isinstance(memory, dict) and str(memory.get("memory") or "").strip()
    ]


def _memory_result_ids(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    memories = result.get("results")
    if not isinstance(memories, list):
        return []
    return [
        str(memory["id"])
        for memory in memories
        if isinstance(memory, dict) and memory.get("id") is not None
    ]


@contextmanager
def _mem0_observation_date(created_at: str) -> Iterator[None]:
    """Supply the source date omitted by mem0ai 2.0.11's OSS add path."""
    _install_mem0_observation_date_wrapper()
    token = _MEM0_OBSERVATION_DATE.set(created_at)
    try:
        yield
    finally:
        _MEM0_OBSERVATION_DATE.reset(token)


def _install_mem0_observation_date_wrapper() -> None:
    """Install one process-wide wrapper whose timestamp is context-local."""
    os.environ.setdefault("MEM0_DIR", default_mem0_dir_string())
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    importlib.import_module("mem0")
    mem0_main = importlib.import_module("mem0.memory.main")

    with _MEM0_PROMPT_PATCH_LOCK:
        current = mem0_main.generate_additive_extraction_prompt
        if getattr(current, "_locomo_observation_date_wrapper", False):
            return

        def with_timestamp(*args: Any, **kwargs: Any) -> Any:
            created_at = _MEM0_OBSERVATION_DATE.get()
            if created_at is not None:
                kwargs.setdefault("timestamp", created_at)
            return current(*args, **kwargs)

        with_timestamp._locomo_observation_date_wrapper = True  # type: ignore[attr-defined]
        mem0_main.generate_additive_extraction_prompt = with_timestamp
