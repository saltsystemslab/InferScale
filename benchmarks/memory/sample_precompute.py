"""Encode one sample's fact chunks for KV injection, through the chunk cache when enabled.

The HF encoder cannot share the GPU with vLLM, so every sample is encoded
before vLLM starts. A cache hit builds the encoder from the cached RoPE tables
and never loads the model weights.
"""

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
    """The settings the precompute reads, as both benchmark configs expose them."""

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
    """Builds the composer of each sample, encoding or loading its chunks."""

    def __init__(self, config: KVPrecomputeSettings) -> None:
        self.config = config
        self._encoder: ChunkedRopeEncoder | None = None
        self._standalone_tokenizer: Any | None = None

    def precompute(
        self, sample: ConversationSample, facts: Sequence[SearchHit]
    ) -> tuple[SampleComposer, bool]:
        """The sample's composer with its chunks in the store, and whether the cache served it."""
        kv_facts = unique_memory_facts(facts)
        chunk_store = self._build_chunk_store()
        key = None
        composer = None
        if self.config.kv_chunk_cache_enabled:
            key = ChunkCacheKey.from_config(self.config, sample, kv_facts)
            composer = self._load_cached_composer(key, sample, facts, kv_facts, chunk_store)
        if composer is not None:
            return composer, True
        return self._encode_composer(sample, facts, chunk_store, key), False

    def release_model(self) -> None:
        """Drop the HF weights before vLLM starts; the tokenizer stays usable."""
        if self._encoder is not None:
            self._encoder.release_model()

    def close(self) -> None:
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None

    def _load_cached_composer(
        self,
        key: ChunkCacheKey,
        sample: ConversationSample,
        facts: Sequence[SearchHit],
        kv_facts: list[RetrievedFact],
        chunk_store: ChunkStore,
    ) -> SampleComposer | None:
        """A composer over the cached encode, or None when the cache misses."""
        path = key.path(self.config.cache_root)
        cached = load_sample_chunks(
            path,
            device=self._chunk_device(),
            scaffold_device=self.config.kv_device,
            expected_meta=key.meta(),
            expected_fact_ids=[fact.memory_id for fact in kv_facts],
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
        encoder = ChunkedRopeEncoder.from_tables(
            model=self.config.model,
            device=self.config.kv_device,
            max_position=self.config.kv_max_position,
            tokenizer=tokenizer,
            cos_table=cached.cos_table,
            sin_table=cached.sin_table,
        )
        logger.info("KV chunk cache hit sample_id={} ({})", sample.sample_id, path.name)
        return SampleComposer.from_cached(
            encoder=encoder,
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
        key: ChunkCacheKey | None,
    ) -> SampleComposer:
        """Encode the sample's facts with the HF encoder, saving the cache when enabled."""
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
        if key is not None:
            self._save_chunk_cache(key, composer, encoder)
        # The disk save needs the real tensors; the store owns them from here
        # (D2H into pinned buffers under the cpu backend).
        composer.move_chunks_to_store()
        return composer

    def _save_chunk_cache(
        self, key: ChunkCacheKey, composer: SampleComposer, encoder: ChunkedRopeEncoder
    ) -> None:
        path = key.path(self.config.cache_root)
        try:
            save_sample_chunks(
                path,
                meta=key.meta(),
                fact_chunks=composer.chunks,
                scaffold_chunks=composer.scaffold_chunks(),
                cos_table=encoder.cos_table,
                sin_table=encoder.sin_table,
            )
        except Exception as exc:
            # The cache is an accelerator, never a correctness dependency; a
            # full disk must not kill the run.
            logger.warning("Failed to save KV chunk cache {}: {}", path, exc)

    def _build_chunk_store(self) -> ChunkStore:
        return build_chunk_store(
            self.config.kv_store_backend,
            device=self.config.kv_device,
            top_k=self.config.top_k,
            staging_slots=self.config.kv_staging_slots,
        )

    def _chunk_device(self) -> str:
        # Under the cpu backend the disk cache loads straight to host and the
        # store pins it, so the payload never touches the GPU.
        return "cpu" if self.config.kv_store_backend == "cpu" else self.config.kv_device

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
        """The encoder-side tokenizer without forcing an HF weight load.

        Reuses the live encoder's tokenizer when one exists; otherwise loads
        the tokenizer standalone (memoized) so fully cached runs never touch
        the model weights.
        """
        if self._encoder is not None and self._encoder.tokenizer is not None:
            return self._encoder.tokenizer
        if self._standalone_tokenizer is None:
            self._standalone_tokenizer = load_encoder_tokenizer(self.config.model)
        return self._standalone_tokenizer
