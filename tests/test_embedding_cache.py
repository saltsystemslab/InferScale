from __future__ import annotations

from locomo_jasper_bench.embedding_cache import CachedEmbedder


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = []

    def embed(self, text, *args, **kwargs):
        self.calls.append({"text": text, "args": args, "kwargs": kwargs})
        return [1.0, 2.0, 3.0]


def test_cached_embedder_writes_and_reuses_npy_cache(tmp_path):
    wrapped = FakeEmbedder()
    embedder = CachedEmbedder(wrapped, cache_dir=tmp_path, model="text-embedding/test")

    first = embedder.embed("hello", "search")
    second = embedder.embed("hello", "search")

    assert first == [1.0, 2.0, 3.0]
    assert second == [1.0, 2.0, 3.0]
    assert len(wrapped.calls) == 1
    assert embedder.stats()["hits"] == 1
    assert embedder.stats()["misses"] == 1
    assert list(embedder.cache_dir.glob("*.npy"))


def test_cached_embedder_keys_include_purpose(tmp_path):
    wrapped = FakeEmbedder()
    embedder = CachedEmbedder(wrapped, cache_dir=tmp_path, model="text-embedding-3-small")

    embedder.embed("hello", "search")
    embedder.embed("hello", "add")

    assert len(wrapped.calls) == 2
    assert embedder.stats()["hits"] == 0
    assert embedder.stats()["misses"] == 2
    assert len(list(embedder.cache_dir.glob("*.npy"))) == 2
