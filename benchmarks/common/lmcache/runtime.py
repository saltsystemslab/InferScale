"""What an answer client does around the LMCache engine once vLLM has started.

Everything here imports lmcache, vllm, and torch inside the functions: the
benchmark modules that use it must import on machines without a GPU.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from inferscale.v1.units import bytes_to_mb

# Each request also stores its unique footer-plus-question tail segment; this
# many tokens per request are budgeted when the exact tails are not known.
TAIL_ALLOWANCE_TOKENS = 128
# The share of the configured CPU cache the warm-up and the tails may fill.
CACHE_FILL_FRACTION = 0.9
GB = 1_000_000_000


@dataclass(slots=True, frozen=True)
class WarmupStats:
    prompt_count: int
    token_count: int
    time_ms: float


def register_model(llm: Any) -> None:
    """Hand the loaded model to LMCache so the deferred blender can be built."""
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
    """The separator token ids LMCache splits prompts on, which the prompts must use.

    LMCache encodes ``blend_special_str`` with its own Hugging Face tokenizer, and
    for Mistral that encoding differs from vLLM's tokenizer, so the ids are read
    from LMCache rather than derived again.
    """
    database = getattr(lmcache_engine(), "token_database", None)
    separator = getattr(database, "sep_tokens", None)
    if separator is None:
        raise RuntimeError("LMCache is not splitting prompts into segments; blending is off.")
    token_ids = [int(token) for token in separator.tolist()]
    if not token_ids:
        raise RuntimeError("LMCache's blend separator encodes to no tokens.")
    return token_ids


def clear_cache() -> int:
    """Empty every LMCache tier; returns the number of removed objects."""
    return int(lmcache_engine().clear())


def cpu_cache_usage_bytes() -> int | None:
    from lmcache.observability import LMCStatsMonitor

    usage = getattr(LMCStatsMonitor.GetOrCreate(), "local_cache_usage_bytes", None)
    return None if usage is None else int(usage)


def kv_bytes_per_token() -> int:
    """Bytes of KV one token occupies in LMCache's cache, from the engine's own layout."""
    import torch

    metadata = lmcache_engine().metadata
    # kv_shape is (layers, 1 or 2, chunk_size, kv_heads, head_size); axis 2 is per token.
    shape = tuple(int(dim) for dim in metadata.kv_shape)
    per_token_elements = math.prod(shape) // shape[2]
    return per_token_elements * torch.tensor([], dtype=metadata.kv_dtype).element_size()


def require_cache_capacity(
    *,
    warm_tokens: int,
    tail_tokens: int,
    bytes_per_token: int,
    max_local_cpu_size_gb: float,
    what: str,
) -> int:
    """Fail before warming when the cache cannot hold the units plus the request tails.

    Eviction would silently turn blend hits into misses, so the limit is enforced
    up front rather than observed afterwards. Returns the required bytes.
    """
    required = int(bytes_per_token) * (int(warm_tokens) + int(tail_tokens))
    limit = max_local_cpu_size_gb * GB * CACHE_FILL_FRACTION
    if required > limit:
        raise RuntimeError(
            f"{what} needs about {required / GB:.1f} GB of KV in LMCache's CPU cache, but "
            f"lmcache.max_local_cpu_size_gb={max_local_cpu_size_gb:g} leaves {limit / GB:.1f} GB "
            "for it; eviction would silently destroy blend hits. Raise max_local_cpu_size_gb."
        )
    return required


def warm_prompts(llm: Any, prompts: Sequence[Sequence[int]], *, batch_size: int) -> WarmupStats:
    """Prefill each prompt once so LMCache stores its segment; one token is generated."""
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
    """The setup metrics every arm records after warming LMCache."""
    usage = cpu_cache_usage_bytes()
    return {
        "lmcache_warm_time_ms": stats.time_ms,
        "lmcache_warm_prompts": stats.prompt_count,
        "lmcache_warm_tokens": stats.token_count,
        "lmcache_cpu_cache_mb": None if usage is None else bytes_to_mb(usage),
    }


def shutdown() -> None:
    """Destroy the LMCache engine and drop the seeded blender so a later engine starts clean."""
    try:
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
    except ImportError:
        return
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    LMCBlenderBuilder._blenders.pop(ENGINE_NAME, None)
