from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from inferscale.v1.units import bytes_to_mb

TAIL_ALLOWANCE_TOKENS = 128
CACHE_FILL_FRACTION = 0.9
GB = 1_000_000_000
CHUNK_SIZE_AXIS = 2
LOCAL_CPU_BACKEND = "LocalCPUBackend"


@dataclass(slots=True, frozen=True)
class WarmupStats:
    prompt_count: int
    token_count: int
    time_ms: float


def register_model(llm: Any) -> None:
    llm.apply_model(_register_with_lmcache)


def _register_with_lmcache(model: Any) -> None:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.compute.models.utils import VLLMModelTracker

    VLLMModelTracker.register_model(ENGINE_NAME, model)


def lmcache_engine() -> Any:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if engine is None:
        raise RuntimeError(
            "The LMCache engine is not initialized; start vLLM with the blend connector first."
        )
    return engine


def lmcache_separator() -> list[int]:
    database = getattr(lmcache_engine(), "token_database", None)
    separator = getattr(database, "sep_tokens", None)
    if separator is None:
        raise RuntimeError("LMCache is not splitting prompts into segments; blending is off.")
    token_ids = [int(token) for token in separator.tolist()]
    if not token_ids:
        raise RuntimeError("LMCache's blend separator encodes to no tokens.")
    return token_ids


def clear_cache() -> int:
    return int(lmcache_engine().clear())


def cpu_cache_usage_bytes() -> int:
    cpu_cache = lmcache_engine().storage_manager.storage_backends[LOCAL_CPU_BACKEND]
    with cpu_cache.cpu_lock:
        return sum(memory_obj.get_size() for memory_obj in cpu_cache.hot_cache.values())


def kv_bytes_per_token() -> int:
    import torch

    metadata = lmcache_engine().metadata
    kv_shape = tuple(int(dim) for dim in metadata.kv_shape)
    per_token_elements = math.prod(kv_shape) // kv_shape[CHUNK_SIZE_AXIS]
    return per_token_elements * torch.tensor([], dtype=metadata.kv_dtype).element_size()


def require_cache_capacity(
    *,
    warm_tokens: int,
    tail_tokens: int,
    bytes_per_token: int,
    max_local_cpu_size_gb: float,
    what: str,
) -> int:
    required_bytes = int(bytes_per_token) * (int(warm_tokens) + int(tail_tokens))
    fillable_bytes = max_local_cpu_size_gb * GB * CACHE_FILL_FRACTION
    if required_bytes > fillable_bytes:
        raise RuntimeError(
            f"{what} needs about {required_bytes / GB:.1f} GB of KV in LMCache's CPU cache, but "
            f"lmcache.max_local_cpu_size_gb={max_local_cpu_size_gb:g} leaves "
            f"{fillable_bytes / GB:.1f} GB for it; eviction would silently destroy blend hits. "
            "Raise max_local_cpu_size_gb."
        )
    return required_bytes


def warm_prompts(llm: Any, prompts: Sequence[Sequence[int]], *, batch_size: int) -> WarmupStats:
    from vllm import SamplingParams

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    params = SamplingParams(max_tokens=1, temperature=0.0)
    started = time.perf_counter()
    for start in range(0, len(prompts), batch_size):
        batch = [
            {"prompt_token_ids": list(prompt)} for prompt in prompts[start : start + batch_size]
        ]
        llm.generate(batch, params, use_tqdm=False)
    return WarmupStats(
        prompt_count=len(prompts),
        token_count=sum(len(prompt) for prompt in prompts),
        time_ms=(time.perf_counter() - started) * 1000,
    )


def warmup_metrics(stats: WarmupStats) -> dict[str, Any]:
    return {
        "lmcache_warm_time_ms": stats.time_ms,
        "lmcache_warm_prompts": stats.prompt_count,
        "lmcache_warm_tokens": stats.token_count,
        "lmcache_cpu_cache_mb": bytes_to_mb(cpu_cache_usage_bytes()),
    }


def shutdown() -> None:
    try:
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
    except ImportError:
        return
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    LMCBlenderBuilder._blenders.pop(ENGINE_NAME, None)
