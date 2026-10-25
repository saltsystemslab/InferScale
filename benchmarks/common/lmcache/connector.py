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

The blender needs five more adjustments for LMCache 0.4.4 on vLLM 0.19. Three
backport LMCache 0.4.7: ``vllm.attention.Attention``, which vLLM moved, is
aliased; the sparse FlashInfer backend, which needs functions this FlashInfer
lacks and is never used here, is stubbed; and the model regains
``get_input_embeddings``, which vLLM renamed. vLLM's Mistral model runs on
LMCache's Llama model, since it subclasses vLLM's Llama. Cached keys move to
their new positions with the model's own RoPE, because LMCache rebuilds RoPE
without scaling and would move Llama 3.1's keys with the wrong frequencies.

Finally, unpinning a cached object that is already unpinned does nothing.
LMCache 0.4.4's layerwise retrieval unpins every object it loads, and the
lookup that pinned them unpins them again, which LMCache answers with a warning
per object and layer inside the timed prefill.
"""

from __future__ import annotations

import importlib
import sys
import time
import types
from typing import Any

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.compute.models.utils import VLLMModelTracker
from vllm.config import set_current_vllm_config
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

    def __init__(self, vllm_config: Any) -> None:
        self._vllm_config = vllm_config
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
            _alias_vllm_attention()
            _stub_sparse_flashinfer_backend()
            _alias_input_embeddings(model)
            _blend_mistral_as_llama()
            # LMCache builds a vLLM rotary embedding for the blender, and vLLM
            # constructs its custom ops only while its config is current.
            with set_current_vllm_config(self._vllm_config):
                blender = LMCBlender(
                    cache_engine=engine,
                    gpu_connector=engine.gpu_connector,
                    vllm_model=model,
                    config=engine.config,
                )
            _move_keys_with_model_rope(blender.layerwise_model, model)
            self._blender = blender
        return self._blender


def _alias_vllm_attention() -> None:
    """Make ``vllm.attention.Attention`` importable for LMCache's attention backends.

    vLLM 0.19 moved the class to ``vllm.model_executor.layers.attention``;
    LMCache 0.4.7 falls back to that path the same way.
    """
    try:
        importlib.import_module("vllm.attention")
    except ModuleNotFoundError:
        from vllm.model_executor.layers.attention.attention import Attention

        module = types.ModuleType("vllm.attention")
        module.Attention = Attention
        sys.modules["vllm.attention"] = module


class _UnavailableSparseBackend:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("LMCache's sparse FlashInfer backend does not run on this FlashInfer.")


def _stub_sparse_flashinfer_backend() -> None:
    """Keep LMCache's sparse FlashInfer backend from breaking its FlashAttention one.

    LMCache 0.4.4 imports both backends together, and the sparse one needs
    FlashInfer functions that vLLM 0.19's FlashInfer lacks. Blending here runs on
    FlashAttention and never enables sparse attention; LMCache 0.4.7 imports each
    backend only when it is used.
    """
    name = "lmcache.v1.compute.attention.flash_infer_sparse"
    try:
        importlib.import_module(name)
    except ImportError:
        module = types.ModuleType(name)
        module.LMCFlashInferSparseBackend = _UnavailableSparseBackend
        sys.modules[name] = module


def _alias_input_embeddings(vllm_model: Any) -> None:
    """Give the model the ``get_input_embeddings`` LMCache 0.4.4 embeds tokens with.

    vLLM 0.18 renamed it to ``embed_input_ids``; LMCache 0.4.7 tries both names.
    """
    if not hasattr(vllm_model, "get_input_embeddings"):
        vllm_model.get_input_embeddings = vllm_model.embed_input_ids


def _blend_mistral_as_llama() -> None:
    """Let LMCache blend Mistral, which vLLM serves with a subclass of its Llama model.

    LMCache 0.4.4 picks its layerwise model by exact class name and knows only
    Llama, Qwen2 and Qwen3. vLLM's ``MistralForCausalLM`` subclasses
    ``LlamaForCausalLM`` and keeps its layer attributes, so LMCache's Llama
    model runs it.
    """
    from lmcache.v1.compute.blend import blender as blender_module
    from lmcache.v1.compute.models.llama import LMCLlamaModel

    infer = blender_module.infer_model_from_vllm
    if getattr(infer, "handles_mistral", False):
        return

    def infer_model_from_vllm(vllm_model: Any, blender: Any, enable_sparse: bool = False) -> Any:
        if type(vllm_model).__name__ == "MistralForCausalLM":
            return LMCLlamaModel(vllm_model, blender, enable_sparse)
        return infer(vllm_model, blender, enable_sparse)

    infer_model_from_vllm.handles_mistral = True
    blender_module.infer_model_from_vllm = infer_model_from_vllm


def _move_keys_with_model_rope(layerwise_model: Any, vllm_model: Any) -> None:
    """Move cached keys to their positions in the prompt with the model's own RoPE.

    LMCache 0.4.4 rebuilds the RoPE for this from the base frequency alone, so it
    would move Llama 3.1's keys, which its scaled RoPE encoded, with the wrong
    frequencies. The model's rotary embedding holds the table that encoded them.
    """
    from lmcache.v1.compute.positional_encoding import FusedRope

    rotary_emb = vllm_model.model.layers[0].self_attn.rotary_emb
    if rotary_emb.rotary_dim != rotary_emb.head_size:
        raise RuntimeError("LMCache blending supports only RoPE over the whole attention head.")
    layerwise_model.fused_rotary_emb = FusedRope(rotary_emb, rotary_emb.is_neox_style)


class _ListTokenRequest:
    """A vLLM request whose ``all_token_ids`` is a plain list; the rest is the request's."""

    def __init__(self, request: Any) -> None:
        self._request = request

    @property
    def all_token_ids(self) -> list[int]:
        return list(self._request.all_token_ids)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request, name)


def seed_deferred_blender(vllm_config: Any) -> None:
    """Make ``LMCBlenderBuilder.get_or_create`` return a deferred blender for this engine."""
    if ENGINE_NAME not in LMCBlenderBuilder._blenders:
        LMCBlenderBuilder._blenders[ENGINE_NAME] = DeferredBlender(vllm_config)


def _skip_repeated_unpin() -> None:
    """Make unpinning a cached object that is already unpinned a no-op.

    LMCache 0.4.4 would take the pin count below zero, log a warning, and reset
    it to zero; it would also free the object a second time if nothing else
    held it. Skipping the call keeps the count and the memory as they were.
    """
    from lmcache.v1.memory_management import TensorMemoryObj

    unpin = TensorMemoryObj.unpin
    if getattr(unpin, "skips_repeats", False):
        return

    def unpin_once(self: Any) -> bool:
        if self.meta.pin_count <= 0:
            return True
        return unpin(self)

    unpin_once.skips_repeats = True
    TensorMemoryObj.unpin = unpin_once


class BlendConnector(LMCacheConnectorV1):
    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None) -> None:
        seed_deferred_blender(vllm_config)
        _skip_repeated_unpin()
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
