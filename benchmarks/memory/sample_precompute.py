from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from inferscale.v1.kv.chunk_store import build_chunk_store
from inferscale.v1.kv.encoder import ChunkedRopeEncoder, load_encoder_tokenizer
from inferscale.v1.protocols import ChunkStore
from inferscale.v1.types import SearchHit

from .chunk_cache import (
    ChunkCacheKey,
    KVEncodingSettings,
    load_sample_chunks,
    save_sample_chunks,
    scaffold_chunks_match,
)
from .composer import SampleComposer
from .data import ConversationSample
from .facts import unique_memory_facts
from .prompting import extract_memory_scaffold_token_ids
from .types import RetrievedFact


class KVPrecomputeSettings(KVEncodingSettings, Protocol):
    @property
    def cache_root(self) -> Path: ...

    @property
    def kv_chunk_cache_enabled(self) -> bool: ...

    @property
    def kv_device(self) -> str: ...

    @property
    def kv_store_backend(self) -> str: ...

    @property
    def kv_staging_slots(self) -> int: ...

    @property
    def top_k(self) -> int: ...


class SamplePrecompute:
    def __init__(self, config: KVPrecomputeSettings) -> None:
        self.config = config
        self._encoder: ChunkedRopeEncoder | None = None
        self._standalone_tokenizer: Any | None = None

    def precompute(
        self, sample: ConversationSample, facts: Sequence[SearchHit]
    ) -> tuple[SampleComposer, bool]:
        unique_facts = unique_memory_facts(facts)
        chunk_store = self._build_chunk_store()
        cache_key = None
        cached_composer = None
        if self.config.kv_chunk_cache_enabled:
            cache_key = ChunkCacheKey.from_config(self.config, sample, unique_facts)
            cached_composer = self._load_cached_composer(
                cache_key, sample, facts, unique_facts, chunk_store
            )
        if cached_composer is not None:
            return cached_composer, True
        return self._encode_composer(sample, facts, chunk_store, cache_key), False

    def release_model(self) -> None:
        if self._encoder is not None:
            self._encoder.release_model()

    def close(self) -> None:
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None

    def _load_cached_composer(
        self,
        cache_key: ChunkCacheKey,
        sample: ConversationSample,
        facts: Sequence[SearchHit],
        unique_facts: list[RetrievedFact],
        chunk_store: ChunkStore,
    ) -> SampleComposer | None:
        cache_path = cache_key.path(self.config.cache_root)
        cached = load_sample_chunks(
            cache_path,
            device=self._cached_chunk_device(),
            scaffold_device=self.config.kv_device,
            expected_meta=cache_key.meta(),
            expected_fact_ids=[fact.memory_id for fact in unique_facts],
        )
        if cached is None:
            return None
        tokenizer = self._encoder_tokenizer()
        scaffold = extract_memory_scaffold_token_ids(
            tokenizer, block_size=self.config.kv_block_size
        )
        if not scaffold_chunks_match(cached.scaffold_chunks, scaffold):
            logger.warning(
                "KV chunk cache scaffold mismatch for sample_id={}; re-encoding", sample.sample_id
            )
            return None
        compose_only_encoder = ChunkedRopeEncoder.from_tables(
            model=self.config.model,
            device=self.config.kv_device,
            max_position=self.config.kv_max_position,
            tokenizer=tokenizer,
            cos_table=cached.cos_table,
            sin_table=cached.sin_table,
        )
        logger.info("KV chunk cache hit sample_id={} ({})", sample.sample_id, cache_path.name)
        return SampleComposer.from_cached(
            encoder=compose_only_encoder,
            cached=cached,
            sample=sample,
            facts=facts,
            context_window=self.config.context_window,
            block_size=self.config.kv_block_size,
            chunk_store=chunk_store,
        )

    def _encode_composer(
        self,
        sample: ConversationSample,
        facts: Sequence[SearchHit],
        chunk_store: ChunkStore,
        cache_key: ChunkCacheKey | None,
    ) -> SampleComposer:
        encoder = self._ensure_encoder()
        composer = SampleComposer(
            encoder=encoder,
            context_window=self.config.context_window,
            block_size=self.config.kv_block_size,
            chunk_store=chunk_store,
        )
        try:
            composer.encode_sample(sample, facts)
        except RuntimeError as exc:
            composer.close()
            raise RuntimeError(f"KV precompute failed for sample_id={sample.sample_id}.") from exc
        except BaseException:
            composer.close()
            raise
        if cache_key is not None:
            self._save_chunk_cache(cache_key, composer, encoder)
        composer.move_chunks_to_store()
        return composer

    def _save_chunk_cache(
        self, cache_key: ChunkCacheKey, composer: SampleComposer, encoder: ChunkedRopeEncoder
    ) -> None:
        cache_path = cache_key.path(self.config.cache_root)
        try:
            save_sample_chunks(
                cache_path,
                meta=cache_key.meta(),
                fact_chunks=composer.chunks,
                scaffold_chunks=composer.scaffold_chunks(),
                cos_table=encoder.cos_table,
                sin_table=encoder.sin_table,
            )
        except Exception as tolerated_failure:
            logger.warning("Failed to save KV chunk cache {}: {}", cache_path, tolerated_failure)

    def _build_chunk_store(self) -> ChunkStore:
        return build_chunk_store(
            self.config.kv_store_backend,
            device=self.config.kv_device,
            top_k=self.config.top_k,
            staging_slots=self.config.kv_staging_slots,
        )

    def _cached_chunk_device(self) -> str:
        store_pins_host_memory = self.config.kv_store_backend == "cpu"
        return "cpu" if store_pins_host_memory else self.config.kv_device

    def _ensure_encoder(self) -> ChunkedRopeEncoder:
        if self._encoder is None:
            self._encoder = ChunkedRopeEncoder(
                model=self.config.model,
                dtype=self.config.kv_dtype,
                device=self.config.kv_device,
                max_position=self.config.kv_max_position,
            )
        return self._encoder

    def _encoder_tokenizer(self) -> Any:
        if self._encoder is not None and self._encoder.tokenizer is not None:
            return self._encoder.tokenizer
        if self._standalone_tokenizer is None:
            self._standalone_tokenizer = load_encoder_tokenizer(self.config.model)
        return self._standalone_tokenizer
