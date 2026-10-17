"""The precompute-kv stage: fill the KV chunk cache before the benchmark runs.

One model and context window per invocation. Requires the Mem0 fact catalogs
from the extract-facts stage; needs no vLLM server because the encoder runs
in-process.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from inferscale.v1.kv.encoder import ChunkedRopeEncoder
from inferscale.v1.types import SearchHit

from .chunk_cache import ChunkCacheKey, save_sample_chunks
from .composer import SampleComposer
from .config import MemoryRunConfig
from .data import ConversationSample, load_locomo
from .facts import unique_memory_facts
from .mem0.fact_catalog import FactCatalogStore, fact_catalog_hits
from .mem0.memory_builder import fact_catalog_store_for


def precompute_kv_chunks(config: MemoryRunConfig) -> dict[str, Any]:
    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    if not samples:
        raise RuntimeError(f"No LoCoMo samples found in {config.dataset_path}.")
    store = fact_catalog_store_for(config)
    encoder: ChunkedRopeEncoder | None = None
    encoded = 0
    skipped = 0
    cache_dir = None
    try:
        for sample in samples:
            hits = _catalog_hits(store, sample)
            key = ChunkCacheKey.from_config(config, sample, unique_memory_facts(hits))
            cache_dir = key.directory(config.cache_root)
            path = key.path(config.cache_root)
            if path.exists():
                skipped += 1
                logger.info("kv-chunks: {} already cached ({})", sample.sample_id, path.name)
                continue
            if encoder is None:
                encoder = _encoder(config)
            _encode_and_save(encoder, config, sample, hits, key)
            encoded += 1
    finally:
        if encoder is not None:
            encoder.close()
    return {
        "samples": len(samples),
        "encoded": encoded,
        "skipped": skipped,
        "cache_dir": str(cache_dir),
    }


def _catalog_hits(store: FactCatalogStore, sample: ConversationSample) -> list[SearchHit]:
    catalog_path = store.path_for(sample)
    if not catalog_path.exists():
        raise RuntimeError(
            f"No Mem0 fact catalog for sample {sample.sample_id} at {catalog_path}. "
            "Extraction always uses the answer model; materialize catalogs with "
            "the extract-facts stage for this model."
        )
    return fact_catalog_hits(store.load(sample))


def _encoder(config: MemoryRunConfig) -> ChunkedRopeEncoder:
    return ChunkedRopeEncoder(
        model=config.model,
        dtype=config.kv_dtype,
        device=config.kv_device,
        max_position=config.kv_max_position,
    )


def _encode_and_save(
    encoder: ChunkedRopeEncoder,
    config: MemoryRunConfig,
    sample: ConversationSample,
    hits: list[SearchHit],
    key: ChunkCacheKey,
) -> None:
    started = time.perf_counter()
    composer = SampleComposer(
        encoder=encoder, context_window=config.context_window, block_size=config.kv_block_size
    )
    try:
        composer.encode_sample(sample, hits)
        fact_count = len(composer.chunks)
        save_sample_chunks(
            key.path(config.cache_root),
            meta=key.meta(),
            fact_chunks=composer.chunks,
            scaffold_chunks=composer.scaffold_chunks(),
            cos_table=encoder.cos_table,
            sin_table=encoder.sin_table,
        )
    finally:
        composer.close()
    logger.info(
        "kv-chunks: encoded {} ({} facts) in {:.1f}s",
        sample.sample_id,
        fact_count,
        time.perf_counter() - started,
    )
