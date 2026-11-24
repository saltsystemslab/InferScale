from __future__ import annotations

from benchmarks.rag.corpus import load_rag_corpus
from benchmarks.rag.preembed import preembed_rag_embeddings


def test_preembed_caches_the_chunks_and_leaves_the_queries_to_be_embedded_live(
    multihop_config, embedded_batches
):
    config = multihop_config(embed_batch_size=100)

    summary = preembed_rag_embeddings(config)

    chunk_texts = [chunk.text for chunk in load_rag_corpus(config).chunks]
    assert embedded_batches == [chunk_texts]
    assert summary["chunk_embedding_count"] == summary["cache"]["misses"] == len(chunk_texts)
