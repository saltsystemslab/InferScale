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
    catalog_store = fact_catalog_store_for(config)
    encoder: ChunkedRopeEncoder | None = None
    encoded_count = 0
    skipped_count = 0
    cache_dir = None
    try:
        for sample in samples:
            catalog_hits = _catalog_hits(catalog_store, sample)
            cache_key = ChunkCacheKey.from_config(config, sample, unique_memory_facts(catalog_hits))
            cache_dir = cache_key.directory(config.cache_root)
            cache_path = cache_key.path(config.cache_root)
            if cache_path.exists():
                skipped_count += 1
                logger.info("kv-chunks: {} already cached ({})", sample.sample_id, cache_path.name)
                continue
            if encoder is None:
                encoder = _load_encoder(config)
            _encode_and_save(encoder, config, sample, catalog_hits, cache_key)
            encoded_count += 1
    finally:
        if encoder is not None:
            encoder.close()
    return {
        "samples": len(samples),
        "encoded": encoded_count,
        "skipped": skipped_count,
        "cache_dir": str(cache_dir),
    }


def _catalog_hits(catalog_store: FactCatalogStore, sample: ConversationSample) -> list[SearchHit]:
    catalog_path = catalog_store.path_for(sample)
    if not catalog_path.exists():
        raise RuntimeError(
            f"No Mem0 fact catalog for sample {sample.sample_id} at {catalog_path}. "
            "Extraction always uses the answer model; materialize catalogs with "
            "the extract-facts stage for this model."
        )
    return fact_catalog_hits(catalog_store.load(sample))


def _load_encoder(config: MemoryRunConfig) -> ChunkedRopeEncoder:
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
    catalog_hits: list[SearchHit],
    cache_key: ChunkCacheKey,
) -> None:
    started = time.perf_counter()
    composer = SampleComposer(
        encoder=encoder, context_window=config.context_window, block_size=config.kv_block_size
    )
    try:
        composer.encode_sample(sample, catalog_hits)
        fact_count = len(composer.chunks)
        save_sample_chunks(
            cache_key.path(config.cache_root),
            meta=cache_key.meta(),
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
