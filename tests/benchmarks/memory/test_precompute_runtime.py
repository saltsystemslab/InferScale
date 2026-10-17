from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.common.config import RuntimeConfig
from benchmarks.memory import precompute_kv
from benchmarks.memory.chunk_cache import ChunkCacheKey
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample


def test_precompute_reuses_cache_from_selected_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeConfig.from_dict({"storage": {"cache_root": str(tmp_path / "cache")}})
    config = MemoryRunConfig.from_dict({"model": "test/model", "inferscale": {}}, runtime=runtime)
    sample = ConversationSample(sample_id="sample", turns=[], qa=[], raw={})
    catalog_path = tmp_path / "catalog.json"
    catalog_path.touch()
    store = SimpleNamespace(path_for=lambda _: catalog_path, load=lambda _: [])
    monkeypatch.setattr(precompute_kv, "load_locomo", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(precompute_kv, "fact_catalog_store_for", lambda _: store)
    path = ChunkCacheKey.from_config(config, sample, []).path(config.cache_root)
    path.parent.mkdir(parents=True)
    path.touch()

    result = precompute_kv.precompute_kv_chunks(config)

    assert result["skipped"] == 1
    assert result["encoded"] == 0
    assert result["cache_dir"] == str(path.parent)
