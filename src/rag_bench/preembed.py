from __future__ import annotations

from typing import Any

from loguru import logger

from locomo_jasper_bench.results import write_json

from .chunking import chunk_corpus
from .config import RagBenchConfig
from .datasets import get_dataset
from .embedder import CHUNK_EMBED_PURPOSE, QUERY_EMBED_PURPOSE, build_cached_embedder
from .tokenizer import load_rag_tokenizer


def preembed_rag_embeddings(config: RagBenchConfig) -> dict[str, Any]:
    """Warm the embedding cache for every corpus chunk and every query.

    Chunk texts are decoded from the answer model's token ids, so preembedding
    is per answer model; the cache itself is content-addressed and shared.
    """
    spec = get_dataset(config.dataset_name)
    docs, queries = spec.load(config.data_dir)
    tokenizer = load_rag_tokenizer(config.model)
    chunks = chunk_corpus(docs, tokenizer=tokenizer, chunk_size=config.chunk_size)
    logger.info(
        "Preembedding dataset={} model={} docs={} chunks={} queries={} cache_dir={}",
        config.dataset_name,
        config.model,
        len(docs),
        len(chunks),
        len(queries),
        config.embedding_cache_dir,
    )

    embedder = build_cached_embedder(config, mode="write")
    _embed_in_batches(
        embedder,
        [chunk.text for chunk in chunks],
        purpose=CHUNK_EMBED_PURPOSE,
        batch_size=config.embed_batch_size,
        label="chunks",
    )
    _embed_in_batches(
        embedder,
        [query.question for query in queries],
        purpose=QUERY_EMBED_PURPOSE,
        batch_size=config.embed_batch_size,
        label="queries",
    )

    summary = {
        "run_id": config.run_id,
        "mode": "rag-preembed",
        "dataset": config.dataset_name,
        "doc_count": len(docs),
        "chunk_embedding_count": len(chunks),
        "query_embedding_count": len(queries),
        "cache": embedder.stats(),
        "config": config.to_jsonable(),
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "preembedding.json", summary)
    logger.info(
        "Preembedded chunks={} queries={} cache_hits={} cache_misses={}",
        len(chunks),
        len(queries),
        embedder.hits,
        embedder.misses,
    )
    return summary


def _embed_in_batches(
    embedder: Any,
    texts: list[str],
    *,
    purpose: str,
    batch_size: int,
    label: str,
) -> None:
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embedder.embed_batch(batch, purpose)
        done = min(start + batch_size, len(texts))
        logger.info("Embedded {}/{} {} (purpose={})", done, len(texts), label, purpose)
