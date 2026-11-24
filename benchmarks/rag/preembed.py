from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.common.embedding_cache import CachedEmbedder
from benchmarks.common.files import write_json
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.corpus import load_rag_corpus
from benchmarks.rag.embedder import build_cached_embedder

PREEMBEDDING_FILE = "preembedding.json"


def preembed_rag_embeddings(config: RagBenchConfig) -> dict[str, Any]:
    corpus = load_rag_corpus(config)
    logger.info(
        "Preembedding dataset={} model={} docs={} chunks={} cache_dir={}",
        config.dataset_name,
        config.model,
        len(corpus.docs),
        len(corpus.chunks),
        config.embedding_cache_dir,
    )
    embedder = build_cached_embedder(config, mode="write")
    _embed_in_batches(
        embedder, [chunk.text for chunk in corpus.chunks], batch_size=config.embed_batch_size
    )
    summary = {
        "run_id": config.run_id,
        "mode": "rag-preembed",
        "dataset": config.dataset_name,
        "doc_count": len(corpus.docs),
        "chunk_embedding_count": len(corpus.chunks),
        "cache": embedder.stats(),
        "config": config.to_jsonable(),
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / PREEMBEDDING_FILE, summary)
    logger.info(
        "Preembedded chunks={} cache_hits={} cache_misses={}",
        len(corpus.chunks),
        embedder.hits,
        embedder.misses,
    )
    return summary


def _embed_in_batches(embedder: CachedEmbedder, texts: list[str], *, batch_size: int) -> None:
    for batch_start in range(0, len(texts), batch_size):
        embedder.embed_documents(texts[batch_start : batch_start + batch_size])
        embedded_count = min(batch_start + batch_size, len(texts))
        logger.info("Embedded {}/{} chunks", embedded_count, len(texts))
