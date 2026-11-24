from __future__ import annotations

import pytest

from benchmarks.common.cache_keys import endpoint_cache_key
from benchmarks.common.embedding_cache import CachedEmbedder


@pytest.mark.parametrize(
    "endpoint,cache_key",
    [
        (None, "endpoint-f578047bcc76d86f"),
        ("http://localhost:8000/v1/", "endpoint-cbff7260a780c934"),
    ],
)
def test_endpoint_cache_keys_of_existing_caches_stay_valid(endpoint, cache_key):
    assert endpoint_cache_key(endpoint) == cache_key


class CountingEmbedder:
    def embed(self, text, memory_action=None):
        return [float(len(text)), 0.5]


def test_cached_embedding_files_of_existing_caches_stay_valid(tmp_path):
    cached = CachedEmbedder(CountingEmbedder(), cache_dir=tmp_path, model="m")

    cached.embed("text", "add")

    cache_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert cache_files == [
        tmp_path / "m/32d5d899202522539cdf5b9f4c13109a007c526c3aa8c238814a149d93cd46ba.npy"
    ]
