from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.common.embedding_cache import CachedEmbedder


def vector_for(text: str) -> list[float]:
    return [float(len(text)), 1.0]


class LibraryEmbedder:
    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return [vector_for(text) for text in texts]

    def embed_query(self, text):
        pytest.fail("The cache embeds a query as a one-document batch.")


class Mem0Embedder:
    def __init__(self) -> None:
        self.config = SimpleNamespace(embedding_dims=2)

    def embed(self, text, memory_action=None):
        return vector_for(text)

    def embed_batch(self, texts, memory_action=None):
        return [vector_for(text) for text in texts]


def test_library_embedder_only_receives_the_document_batches_that_are_not_cached(tmp_path: Path):
    wrapped = LibraryEmbedder()
    cached = CachedEmbedder(wrapped, cache_dir=tmp_path, model="model")

    query_vector = cached.embed_query("question")
    document_vectors = cached.embed_documents(["a", "bb", "a"])
    repeated_query_vector = cached.embed_query("question")

    assert isinstance(query_vector, np.ndarray)
    assert (query_vector.dtype, query_vector.tolist()) == (np.float32, [8.0, 1.0])
    assert document_vectors == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]
    assert repeated_query_vector.tolist() == [8.0, 1.0]
    assert wrapped.document_batches == [["question"], ["a", "bb"]]
    assert (cached.hits, cached.misses) == (1, 4)


def test_cache_exposes_the_provider_attributes_that_mem0_reads(tmp_path: Path):
    wrapped = Mem0Embedder()
    cached = CachedEmbedder(wrapped, cache_dir=tmp_path, model="model")

    assert cached.config is wrapped.config
    with pytest.raises(AttributeError, match="no attribute 'collection_name'"):
        _ = cached.collection_name


def test_embedding_time_grows_with_every_call_even_a_failing_one(tmp_path: Path):
    cached = CachedEmbedder(Mem0Embedder(), cache_dir=tmp_path, model="model")
    assert cached.embed_time_ms == 0.0

    cached.embed("fact", "add")
    after_single = cached.embed_time_ms
    cached.embed_batch(["fact", "other"], "add")
    after_batch = cached.embed_time_ms
    with pytest.raises(TypeError):
        cached.embed(5, "add")

    assert 0.0 < after_single < after_batch < cached.embed_time_ms
