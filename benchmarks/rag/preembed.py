"""The preembed stage: warm the embedding cache for every corpus chunk and every query."""

from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.common.embedding_cache import CachedEmbedder
from benchmarks.common.files import write_json
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.corpus import load_rag_corpus
from benchmarks.rag.embedder import CHUNK_EMBED_PURPOSE, QUERY_EMBED_PURPOSE, build_cached_embedder

PREEMBEDDING_FILE = "preembedding.json"


def preembed_rag_embeddings(config: RagBenchConfig) -> dict[str, Any]:
    """Embed every chunk and query once; later stages read the cache and never call the API.

    Chunk texts are decoded from the answer model's token ids, so preembedding
    is per answer model; the cache itself is content-addressed and shared.
    """
    corpus = load_rag_corpus(config)
    logger.info(
        "Preembedding dataset={} model={} docs={} chunks={} queries={} cache_dir={}",
        config.dataset_name,
        config.model,
        len(corpus.docs),
        len(corpus.chunks),
        len(corpus.queries),
        config.embedding_cache_dir,
    )
    embedder = build_cached_embedder(config, mode="write")
    _embed_in_batches(
        embedder,
        [chunk.text for chunk in corpus.chunks],
        purpose=CHUNK_EMBED_PURPOSE,
        batch_size=config.embed_batch_size,
        label="chunks",
    )
    _embed_in_batches(
        embedder,
        [query.question for query in corpus.queries],
        purpose=QUERY_EMBED_PURPOSE,
        batch_size=config.embed_batch_size,
        label="queries",
    )
    summary = {
        "run_id": config.run_id,
        "mode": "rag-preembed",
        "dataset": config.dataset_name,
        "doc_count": len(corpus.docs),
        "chunk_embedding_count": len(corpus.chunks),
        "query_embedding_count": len(corpus.queries),
        "cache": embedder.stats(),
        "config": config.to_jsonable(),
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / PREEMBEDDING_FILE, summary)
    logger.info(
        "Preembedded chunks={} queries={} cache_hits={} cache_misses={}",
        len(corpus.chunks),
        len(corpus.queries),
        embedder.hits,
        embedder.misses,
    )
    return summary


def _embed_in_batches(
    embedder: CachedEmbedder,
    texts: list[str],
    *,
    purpose: str,
    batch_size: int,
    label: str,
) -> None:
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embedder.embed_batch(batch, memory_action=purpose)
        done = min(start + batch_size, len(texts))
        logger.info("Embedded {}/{} {} (purpose={})", done, len(texts), label, purpose)
