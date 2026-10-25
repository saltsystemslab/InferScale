"""vLLM's LMCache connector, adapted for in-process CacheBlend measurements.

``BlendConnector`` is loaded by module path (``kv_connector_module_path``), so
it carries a name vLLM has not registered. It changes four things about the
connector it subclasses: the blender is created on first use instead of at
construction, because vLLM builds the connector before the model is loaded
(LMCache issue 4131); CUDA graphs always run piecewise, because blending is
layerwise; LMCache sees each request's token ids as a plain list, because its
blending lookup sends them to the lookup server with msgpack, which cannot
encode vLLM's read-only ``ConstantList``; and each request's matched token
count and load time are recorded for the benchmark (see ``load_stats``).
"""

from __future__ import annotations

import time
from typing import Any

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.compute.models.utils import VLLMModelTracker
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import LMCacheConnectorV1

from .load_stats import load_stats, reset_load_stats, snapshot_load_stats

__all__ = [
    "BlendConnector",
    "DeferredBlender",
    "reset_load_stats",
    "seed_deferred_blender",
    "snapshot_load_stats",
]


class DeferredBlender:
    """Stands in for ``LMCBlender`` until the model exists, then forwards to it."""

    def __init__(self) -> None:
        self._blender: Any | None = None

    def blend(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve().blend(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def _resolve(self) -> Any:
        if self._blender is None:
            engine = LMCacheEngineBuilder.get(ENGINE_NAME)
            if engine is None:
                raise RuntimeError(
                    "The LMCache engine is not initialized; the blender cannot be built."
                )
            try:
                model = VLLMModelTracker.get_model(ENGINE_NAME)
            except ValueError as exc:
                raise RuntimeError(
                    "The vLLM model is not registered with LMCache; call "
                    "benchmarks.common.lmcache.runtime.register_model(llm) after the engine "
                    "starts and before the first blended request."
                ) from exc
            self._blender = LMCBlender(
                cache_engine=engine,
                gpu_connector=engine.gpu_connector,
                vllm_model=model,
                config=engine.config,
            )
        return self._blender


class _ListTokenRequest:
    """A vLLM request whose ``all_token_ids`` is a plain list; the rest is the request's."""

    def __init__(self, request: Any) -> None:
        self._request = request

    @property
    def all_token_ids(self) -> list[int]:
        return list(self._request.all_token_ids)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request, name)


def seed_deferred_blender() -> None:
    """Make ``LMCBlenderBuilder.get_or_create`` return a deferred blender for this engine."""
    if ENGINE_NAME not in LMCBlenderBuilder._blenders:
        LMCBlenderBuilder._blenders[ENGINE_NAME] = DeferredBlender()


class BlendConnector(LMCacheConnectorV1):
    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None) -> None:
        seed_deferred_blender()
        super().__init__(vllm_config, role, kv_cache_config)

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return True

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> Any:
        # LMCache 0.4.4 passes all_token_ids unchanged to the blending lookup.
        result = super().get_num_new_matched_tokens(_ListTokenRequest(request), num_computed_tokens)
        matched = result[0] if isinstance(result, tuple) else result
        if matched is not None:
            load_stats.record_match(str(request.request_id), int(matched))
        return result

    def start_load_kv(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return super().start_load_kv(*args, **kwargs)
        finally:
            load_stats.record_load_time((time.perf_counter() - started) * 1000)

    def wait_for_layer_load(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return super().wait_for_layer_load(*args, **kwargs)
        finally:
            load_stats.record_load_time((time.perf_counter() - started) * 1000)
