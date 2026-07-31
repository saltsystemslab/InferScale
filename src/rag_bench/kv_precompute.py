from __future__ import annotations

from typing import Any

from loguru import logger

from locomo_jasper_bench.kv.chunked_rope import ChunkedRopeEncoder
from locomo_jasper_bench.kv.vllm_runtime import empty_cuda_cache

from .chunking import chunk_corpus, chunks_by_doc, corpus_fingerprint
from .config import RagBenchConfig
from .datasets import get_dataset
from .kv_cache import (
    cache_meta_base,
    chunk_file_path,
    chunk_meta,
    load_tables_and_scaffold,
    manifest_path,
    rag_chunk_cache_dir,
    rag_scaffold_chunks_match,
    save_chunk,
    save_tables_and_scaffold,
    tables_meta,
    tables_scaffold_path,
    write_manifest,
)
from .kv_plan import build_chunk_context_encoding_plan
from .prompting import extract_rag_scaffold_token_ids
from .tokenizer import load_rag_tokenizer

_EMPTY_CACHE_EVERY = 25


def precompute_rag_kv(config: RagBenchConfig) -> dict[str, Any]:
    """Resumable per-chunk KV precompute into the RAG chunk disk cache.

    Existing chunk files are skipped; interrupt and rerun freely. One
    invocation covers one (model, chunk_size, context_window) configuration.
    """
    spec = get_dataset(config.dataset_name)
    docs, _ = spec.load(config.data_dir)
    tokenizer = load_rag_tokenizer(config.model)
    chunks = chunk_corpus(docs, tokenizer=tokenizer, chunk_size=config.chunk_size)
    grouped = chunks_by_doc(chunks)
    fingerprint = corpus_fingerprint(docs)
    cache_dir = rag_chunk_cache_dir(
        model=config.model,
        dtype=config.kv_dtype,
        chunk_size=config.chunk_size,
        context_window=config.context_window,
        max_position=config.kv_max_position,
        corpus_fingerprint=fingerprint,
        cache_root=config.kv_chunk_cache_root,
    )
    meta_base = cache_meta_base(
        dataset=config.dataset_name,
        model=config.model,
        dtype=config.kv_dtype,
        chunk_size=config.chunk_size,
        context_window=config.context_window,
        max_position=config.kv_max_position,
        corpus_fingerprint=fingerprint,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(cache_dir, meta=meta_base, chunks=chunks)

    missing = [
        chunk for chunk in chunks if not chunk_file_path(cache_dir, chunk.chunk_id).exists()
    ]
    scaffold = extract_rag_scaffold_token_ids(
        tokenizer,
        system_prompt=spec.prompt_profile.system_prompt,
        block_size=config.kv_block_size,
    )
    tables_path = tables_scaffold_path(cache_dir, config.kv_block_size)
    tables_expected_meta = tables_meta(meta_base, block_size=config.kv_block_size)
    cached_tables = load_tables_and_scaffold(
        tables_path, expected_meta=tables_expected_meta, device="cpu"
    )
    need_tables = cached_tables is None or not rag_scaffold_chunks_match(
        cached_tables.scaffold_chunks, scaffold
    )
    logger.info(
        "RAG KV precompute dataset={} model={} chunks={} missing={} tables_needed={} cache_dir={}",
        config.dataset_name,
        config.model,
        len(chunks),
        len(missing),
        need_tables,
        cache_dir,
    )

    encoded = 0
    if missing or need_tables:
        encoder = ChunkedRopeEncoder(
            model=config.model,
            dtype=config.kv_dtype,
            device=config.kv_device,
            max_position=config.kv_max_position,
        )
        try:
            for index, chunk in enumerate(missing, start=1):
                plan = build_chunk_context_encoding_plan(
                    chunk,
                    grouped[chunk.doc_id],
                    context_window=config.context_window,
                    max_input_tokens=config.kv_max_position,
                )
                encoded_chunk = encoder.encode_fact_chunk(plan)
                save_chunk(
                    chunk_file_path(cache_dir, chunk.chunk_id),
                    meta=chunk_meta(meta_base, chunk),
                    chunk=encoded_chunk,
                )
                del encoded_chunk
                encoded += 1
                if index % _EMPTY_CACHE_EVERY == 0 or index == len(missing):
                    empty_cuda_cache()
                    logger.info("Encoded {}/{} missing RAG KV chunks", index, len(missing))
            if need_tables:
                scaffold_chunks = {
                    "header": encoder.encode_token_ids_chunk(
                        "__rag_header__", scaffold.header_token_ids
                    ),
                    "empty_passages": encoder.encode_token_ids_chunk(
                        "__rag_empty_passages__", scaffold.empty_memory_token_ids
                    ),
                    "footer": encoder.encode_token_ids_chunk(
                        "__rag_footer__", scaffold.footer_token_ids
                    ),
                }
                save_tables_and_scaffold(
                    tables_path,
                    meta=tables_expected_meta,
                    cos_table=encoder.cos_table,
                    sin_table=encoder.sin_table,
                    scaffold_chunks=scaffold_chunks,
                )
        finally:
            encoder.close()

    cache_bytes = _cache_bytes(cache_dir)
    summary = {
        "cache_dir": str(cache_dir),
        "manifest": str(manifest_path(cache_dir)),
        "corpus_fingerprint": fingerprint,
        "chunk_count": len(chunks),
        "encoded": encoded,
        "skipped": len(chunks) - len(missing),
        "cache_bytes": cache_bytes,
    }
    logger.info(
        "RAG KV precompute complete: encoded={} skipped={} cache_bytes={}",
        summary["encoded"],
        summary["skipped"],
        cache_bytes,
    )
    return summary


def _cache_bytes(cache_dir) -> int:
    total = 0
    chunks_dir = cache_dir / "chunks"
    if chunks_dir.exists():
        total += sum(path.stat().st_size for path in chunks_dir.glob("*.pt"))
    total += sum(path.stat().st_size for path in cache_dir.glob("tables-scaffold-*.pt"))
    return total
