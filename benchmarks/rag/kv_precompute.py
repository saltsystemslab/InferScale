"""The precompute-kv stage: encode every corpus chunk once into the per-chunk disk cache."""

from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.rag.chunking import chunks_by_doc
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.corpus import RagCorpus, load_rag_corpus
from benchmarks.rag.data_types import RagChunk
from benchmarks.rag.kv_cache import (
    RagChunkCache,
    load_tables_and_scaffold,
    rag_scaffold_chunks_match,
    save_chunk,
    save_tables_and_scaffold,
)
from benchmarks.rag.kv_plan import build_chunk_context_encoding_plan
from benchmarks.rag.prompting import extract_rag_scaffold_token_ids
from inferscale.v1.kv.encoder import ChunkedRopeEncoder
from inferscale.v1.kv.prompt import ScaffoldTokens
from inferscale.v1.serving.vllm import empty_cuda_cache
from inferscale.v1.types import KVChunk

# Releasing the allocator cache after every batch of encodes keeps fragmentation
# from growing across thousands of forward passes.
EMPTY_CACHE_EVERY = 25


def precompute_rag_kv(config: RagBenchConfig) -> dict[str, Any]:
    """Encode the chunks whose files are absent; interrupt and rerun freely.

    One invocation covers one (model, chunk_size, context_window) setting.
    """
    corpus = load_rag_corpus(config)
    cache = RagChunkCache.for_config(config, corpus.docs)
    cache.directory.mkdir(parents=True, exist_ok=True)
    cache.write_manifest(corpus.chunks)
    missing_ids = set(cache.missing_chunk_ids(chunk.chunk_id for chunk in corpus.chunks))
    missing = [chunk for chunk in corpus.chunks if chunk.chunk_id in missing_ids]
    scaffold = extract_rag_scaffold_token_ids(
        corpus.tokenizer,
        system_prompt=corpus.spec.prompt_profile.system_prompt,
        block_size=config.kv_block_size,
    )
    need_tables = not _tables_are_current(cache, config, scaffold)
    logger.info(
        "RAG KV precompute dataset={} model={} chunks={} missing={} tables_needed={} cache_dir={}",
        config.dataset_name,
        config.model,
        len(corpus.chunks),
        len(missing),
        need_tables,
        cache.directory,
    )
    if missing or need_tables:
        encoder = ChunkedRopeEncoder(
            model=config.model,
            dtype=config.kv_dtype,
            device=config.kv_device,
            max_position=config.kv_max_position,
        )
        try:
            _encode_missing_chunks(encoder, config, corpus, cache, missing)
            if need_tables:
                _save_tables(encoder, config, cache, scaffold)
        finally:
            encoder.close()
    summary = {
        "cache_dir": str(cache.directory),
        "manifest": str(cache.manifest_path),
        "corpus_fingerprint": corpus.fingerprint,
        "chunk_count": len(corpus.chunks),
        "encoded": len(missing),
        "skipped": len(corpus.chunks) - len(missing),
        "cache_bytes": cache.total_bytes(),
    }
    logger.info(
        "RAG KV precompute complete: encoded={} skipped={} cache_bytes={}",
        summary["encoded"],
        summary["skipped"],
        summary["cache_bytes"],
    )
    return summary


def _tables_are_current(
    cache: RagChunkCache, config: RagBenchConfig, scaffold: ScaffoldTokens
) -> bool:
    cached = load_tables_and_scaffold(
        cache.tables_path(config.kv_block_size),
        expected_meta=cache.key.tables_meta(block_size=config.kv_block_size),
        device="cpu",
    )
    return cached is not None and rag_scaffold_chunks_match(cached.scaffold_chunks, scaffold)


def _encode_missing_chunks(
    encoder: ChunkedRopeEncoder,
    config: RagBenchConfig,
    corpus: RagCorpus,
    cache: RagChunkCache,
    missing: list[RagChunk],
) -> None:
    grouped = chunks_by_doc(corpus.chunks)
    for index, chunk in enumerate(missing, start=1):
        plan = build_chunk_context_encoding_plan(
            chunk,
            grouped[chunk.doc_id],
            context_window=config.context_window,
            max_input_tokens=config.kv_max_position,
        )
        encoded = encoder.encode_plan(plan)
        save_chunk(
            cache.chunk_path(chunk.chunk_id), meta=cache.key.chunk_meta(chunk), chunk=encoded
        )
        del encoded
        if index % EMPTY_CACHE_EVERY == 0 or index == len(missing):
            empty_cuda_cache()
            logger.info("Encoded {}/{} missing RAG KV chunks", index, len(missing))


def _save_tables(
    encoder: ChunkedRopeEncoder,
    config: RagBenchConfig,
    cache: RagChunkCache,
    scaffold: ScaffoldTokens,
) -> None:
    scaffold_chunks: dict[str, KVChunk] = {
        "header": encoder.encode("__rag_header__", scaffold.header_token_ids),
        "empty_passages": encoder.encode("__rag_empty_passages__", scaffold.empty_token_ids),
        "footer": encoder.encode("__rag_footer__", scaffold.footer_token_ids),
    }
    save_tables_and_scaffold(
        cache.tables_path(config.kv_block_size),
        meta=cache.key.tables_meta(block_size=config.kv_block_size),
        cos_table=encoder.cos_table,
        sin_table=encoder.sin_table,
        scaffold_chunks=scaffold_chunks,
    )
