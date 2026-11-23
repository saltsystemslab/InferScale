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


@pytest.mark.parametrize(
    "endpoint,purpose,text,cache_file",
    [
        (
            None,
            "add",
            "text",
            "m/endpoint-f578047bcc76d86f/"
            "33c0761cac820ca6c49d360afc460766e9aab3b3307eeaf703ff6e1677c1d076.npy",
        ),
        (
            None,
            "search",
            "text",
            "m/endpoint-f578047bcc76d86f/"
            "b04521cb0a0dfacfedcab2d800c0604819926b02df1fb437b6746c84c5b9f89d.npy",
        ),
    ],
)
def test_cached_embedding_files_of_existing_caches_stay_valid(
    tmp_path, endpoint, purpose, text, cache_file
):
    cached = CachedEmbedder(CountingEmbedder(), cache_dir=tmp_path, model="m", endpoint=endpoint)

    cached.embed(text, purpose)

    cache_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert cache_files == [tmp_path / cache_file]
