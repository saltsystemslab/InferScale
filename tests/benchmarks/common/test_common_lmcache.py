from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.common.lmcache import runtime
from benchmarks.common.lmcache.config import (
    ADAPTERS,
    DEFAULT_ADAPTER,
    LMCacheConfig,
    connector_extra_config,
)


def test_package_adapter_ships_with_lmcache_and_vendored_adapter_is_its_copy_native_to_vllm():
    assert (ADAPTERS, DEFAULT_ADAPTER) == (("package", "vendored"), "package")
    assert "use_native" not in connector_extra_config(LMCacheConfig(1.0))
    assert "use_native" not in connector_extra_config(LMCacheConfig(1.0, adapter="package"))
    assert connector_extra_config(LMCacheConfig(1.0, adapter="vendored"))["use_native"] is True


def capacity(**overrides: Any) -> int:
    request = {
        "warm_tokens": 900_000,
        "tail_tokens": 0,
        "bytes_per_token": 1000,
        "max_local_cpu_size_gb": 1,
        "what": "The LoCoMo warm-up",
    }
    return runtime.require_cache_capacity(**{**request, **overrides})


def test_warm_up_and_tails_may_fill_nine_tenths_of_the_cpu_cache():
    assert capacity() == 900_000_000
    assert capacity(warm_tokens=1000, tail_tokens=128) == 1_128_000
    assert capacity(warm_tokens=10.9, tail_tokens=5.9, bytes_per_token=100.9) == 1500

    with pytest.raises(RuntimeError) as too_large:
        capacity(tail_tokens=1)

    assert str(too_large.value) == (
        "The LoCoMo warm-up needs about 0.9 GB of KV in LMCache's CPU cache, but "
        "lmcache.max_local_cpu_size_gb=1 leaves 0.9 GB for it; eviction would silently "
        "destroy blend hits. Raise max_local_cpu_size_gb."
    )


def test_each_request_also_stores_its_footer_plus_question_tail_budgeted_at_128_tokens():
    request_count = 3

    assert runtime.TAIL_ALLOWANCE_TOKENS == 128
    assert capacity(warm_tokens=0, tail_tokens=runtime.TAIL_ALLOWANCE_TOKENS * request_count) == (
        128 * request_count * 1000
    )


def lmcache_engine_with(monkeypatch: pytest.MonkeyPatch, **attributes: Any) -> None:
    monkeypatch.setattr(runtime, "lmcache_engine", lambda: SimpleNamespace(**attributes))


@pytest.mark.parametrize(
    "kv_shape,kv_dtype,bytes_per_token",
    [
        ((32, 2, 256, 8, 128), "bfloat16", 32 * 2 * 8 * 128 * 2),
        ((28, 1, 256, 4, 64), "float16", 28 * 1 * 4 * 64 * 2),
    ],
)
def test_kv_shape_is_layers_one_or_two_tensors_chunk_size_heads_and_head_size(
    monkeypatch, kv_shape, kv_dtype, bytes_per_token
):
    torch = pytest.importorskip("torch")
    metadata = SimpleNamespace(kv_shape=torch.Size(kv_shape), kv_dtype=getattr(torch, kv_dtype))
    lmcache_engine_with(monkeypatch, metadata=metadata)

    assert runtime.kv_bytes_per_token() == bytes_per_token
