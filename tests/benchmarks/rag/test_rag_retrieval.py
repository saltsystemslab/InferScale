from __future__ import annotations

import pytest

from benchmarks.rag.corpus import load_rag_corpus
from benchmarks.rag.embedder import build_cached_embedder
from benchmarks.rag.preembed import preembed_rag_embeddings
from benchmarks.rag.retrieval import CorpusRetriever


@pytest.fixture
def preembedded(multihop_config, embedded_batches):
    config = multihop_config()
    preembed_rag_embeddings(config)
    return config, load_rag_corpus(config)


def offline_retriever(config, chunks) -> CorpusRetriever:
    return CorpusRetriever(
        config=config, chunks=chunks, embedder=build_cached_embedder(config, mode="read")
    )


def test_search_returns_the_ranked_chunks_and_the_retrieval_latency(preembedded, jasper_indexes):
    config, corpus = preembedded
    retriever = offline_retriever(config, corpus.chunks)
    retriever.build()

    hits, metrics = retriever.search(corpus.queries[0].question, top_k=2)

    assert [hit.rank for hit in hits] == [1, 2]
    assert {hit.id for hit in hits} <= set(corpus.chunks_by_id)
    assert hits[0].payload["doc_id"] == corpus.chunks_by_id[hits[0].id].doc_id
    assert metrics.search_time_ms == 0.5
    assert metrics.total_time_ms == metrics.embedding_time_ms + 0.5
    assert metrics.vector_backend == "fake"
