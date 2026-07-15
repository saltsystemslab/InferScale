"""Setup-time KV chunk precompute: fill the chunk cache before benchmark runs.

Driven by `locomo-jasper-bench --precompute-kv-only` (one model and context
window per invocation) and swept by scripts/precompute_kv_chunks.sh. Requires
the Mem0 fact catalogs to exist already (scripts/extract_facts.sh); needs no
vLLM server since the encoder runs in-process.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import BenchmarkConfig
from ..data import load_locomo
from ..retrieval.fact_catalog import fact_catalog_hits
from ..retrieval.memory_builder import fact_catalog_store_for
from .chunk_cache import cache_meta, cache_path_for, chunk_cache_dir, save_sample_chunks
from .context import unique_memory_facts


def precompute_kv_chunks(config: BenchmarkConfig) -> dict[str, Any]:
    from .chunked_rope import ChunkedRopeEncoder, ChunkedRopeSampleComposer

    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    if not samples:
        raise RuntimeError(f"No LoCoMo samples found in {config.dataset_path}.")
    store = fact_catalog_store_for(config)

    encoder: ChunkedRopeEncoder | None = None
    encoded = 0
    skipped = 0
    try:
        for sample in samples:
            catalog_path = store.path_for(sample)
            if not catalog_path.exists():
                raise RuntimeError(
                    f"No Mem0 fact catalog for sample {sample.sample_id} at {catalog_path}. "
                    "Extraction always uses the answer model; materialize catalogs with:\n"
                    f'  EXTRACTION_MODELS="{config.model}" bash scripts/extract_facts.sh'
                )
            hits = fact_catalog_hits(store.load(sample))
            kv_facts = unique_memory_facts(hits)
            path = cache_path_for(
                model=config.model,
                dtype=config.kv_dtype,
                context_window=config.context_window,
                max_position=config.kv_max_position,
                block_size=config.kv_block_size,
                sample=sample,
                facts=kv_facts,
            )
            if path.exists():
                skipped += 1
                print(f"kv-chunks: {sample.sample_id} already cached ({path.name})", flush=True)
                continue
            if encoder is None:
                encoder = ChunkedRopeEncoder(
                    model=config.model,
                    dtype=config.kv_dtype,
                    device=config.kv_device,
                    max_position=config.kv_max_position,
                )
            started = time.perf_counter()
            composer = ChunkedRopeSampleComposer(
                encoder=encoder,
                context_window=config.context_window,
                block_size=config.kv_block_size,
            )
            try:
                composer.encode_sample(sample, hits)
                save_sample_chunks(
                    path,
                    meta=cache_meta(
                        model=config.model,
                        dtype=config.kv_dtype,
                        context_window=config.context_window,
                        max_position=config.kv_max_position,
                        block_size=config.kv_block_size,
                        sample=sample,
                        facts=kv_facts,
                    ),
                    fact_chunks=composer.chunks,
                    scaffold_chunks={
                        "header": composer.header_chunk,
                        "memory_list_header": composer.memory_list_header_chunk,
                        "empty_memory": composer.empty_memory_chunk,
                        "footer": composer.footer_chunk,
                    },
                    cos_table=encoder.cos_table,
                    sin_table=encoder.sin_table,
                )
            finally:
                composer.close()
            encoded += 1
            print(
                f"kv-chunks: encoded {sample.sample_id} "
                f"({len(kv_facts)} facts) in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    finally:
        if encoder is not None:
            encoder.close()

    cache_dir = chunk_cache_dir(
        model=config.model,
        dtype=config.kv_dtype,
        context_window=config.context_window,
        max_position=config.kv_max_position,
        block_size=config.kv_block_size,
    )
    return {
        "samples": len(samples),
        "encoded": encoded,
        "skipped": skipped,
        "cache_dir": str(cache_dir),
    }
