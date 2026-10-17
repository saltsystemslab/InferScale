"""Scatter of a composed memory's KV into vLLM's paged KV cache."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backend import AttentionMetadata

from .connector_metadata import MemoryLoadMeta
from .memory_store import UserMemory

logger = init_logger(__name__)


def inject_load(
    memory: UserMemory, load: MemoryLoadMeta, forward_context: Any, attn_metadata: Any
) -> None:
    """Scatter one load's memory tokens into every attention layer's KV cache."""
    # Probe one layer for the device; the pinned-host view waits on a layer's
    # copy event on first access, so touch exactly one.
    first_layer = next(iter(memory.kv_by_layer), None)
    first_tensor = memory.kv_by_layer[first_layer] if first_layer is not None else None
    if first_tensor is None:
        raise RuntimeError(f"Memory for user {load.user_id} has no layer tensors.")
    slot_mapping = load.slot_mapping.to(device=first_tensor.device, dtype=torch.long)

    injected_layers = 0
    for layer_name, layer in forward_context.no_compile_layers.items():
        # no_compile_layers can hold layers without a paged KV cache.
        kv_cache = getattr(layer, "kv_cache", None)
        if kv_cache is None:
            continue
        src_kv = memory.kv_by_layer.get(layer_name)
        if src_kv is None:
            raise RuntimeError(f"Layer {layer_name} not found in memory for user {load.user_id}.")
        inject_layer(
            dst_kv_cache_layer=kv_cache,
            src_kv_cache=slice_memory_tokens(src_kv, load.skip_tokens, load.num_tokens),
            slot_mapping=slot_mapping,
            attn_metadata=attn_metadata,
            layer_name=layer_name,
        )
        injected_layers += 1
    if injected_layers == 0:
        raise RuntimeError(f"No KV cache layers were injected for user {load.user_id}.")


def slice_memory_tokens(src_kv: torch.Tensor, skip_tokens: int, num_tokens: int) -> torch.Tensor:
    if src_kv.ndim >= 4 and src_kv.shape[0] == 2:
        return src_kv[:, skip_tokens : skip_tokens + num_tokens]
    return src_kv[skip_tokens : skip_tokens + num_tokens]


def inject_layer(
    *,
    dst_kv_cache_layer: torch.Tensor,
    src_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_metadata: AttentionMetadata,
    layer_name: str,
) -> None:
    if isinstance(attn_metadata, dict):
        layer_meta = attn_metadata.get(layer_name, attn_metadata)
    else:
        layer_meta = attn_metadata
    if isinstance(layer_meta, MLACommonMetadata):
        scatter_mla(dst_kv_cache_layer, src_kv_cache, slot_mapping)
        return

    if src_kv_cache.ndim < 4 or src_kv_cache.shape[0] != 2:
        raise ValueError(
            f"Expected source KV for {layer_name} to have shape [2, tokens, ...], "
            f"got {tuple(src_kv_cache.shape)}"
        )
    dst_shape = dst_kv_cache_layer.shape
    if dst_kv_cache_layer.ndim < 4:
        raise ValueError(f"Unexpected paged KV cache shape for {layer_name}: {tuple(dst_shape)}")
    split_dim = kv_split_dim(dst_shape, layer_name)

    page_size = dst_shape[2]
    slots = slot_mapping.to(device=dst_kv_cache_layer.device, dtype=torch.long)
    pages = slots // page_size
    offsets = slots % page_size

    src_k = src_kv_cache[0].to(dtype=dst_kv_cache_layer.dtype, device=dst_kv_cache_layer.device)
    src_v = src_kv_cache[1].to(dtype=dst_kv_cache_layer.dtype, device=dst_kv_cache_layer.device)
    src_k, src_v = match_tail_shape(src_k, src_v, tuple(dst_shape[3:]), layer_name)

    if split_dim == 0:
        dst_kv_cache_layer[0, pages, offsets] = src_k
        dst_kv_cache_layer[1, pages, offsets] = src_v
    else:
        dst_kv_cache_layer[pages, 0, offsets] = src_k
        dst_kv_cache_layer[pages, 1, offsets] = src_v


def scatter_mla(
    dst_kv_cache_layer: torch.Tensor, src_kv_cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    """MLA caches are flat per slot: one latent row per token, no K/V split."""
    pages = dst_kv_cache_layer.shape[0]
    page_size = dst_kv_cache_layer.shape[1]
    dst_flat = dst_kv_cache_layer.reshape(pages * page_size, -1)
    dst_flat[slot_mapping] = src_kv_cache.to(
        dtype=dst_kv_cache_layer.dtype, device=dst_kv_cache_layer.device
    )


def kv_split_dim(dst_shape: Any, layer_name: str) -> int:
    """Axis along which the paged cache separates keys from values."""
    shape_0, shape_1 = dst_shape[0], dst_shape[1]
    if shape_1 == 2 and shape_0 != 2:
        return 1
    if shape_0 == 2 and shape_1 != 2:
        return 0
    if shape_0 == 2 and shape_1 == 2:
        logger.warning_once(
            "Ambiguous KV cache layout for %s shape=%s; defaulting to split dim 1.",
            layer_name,
            tuple(dst_shape),
        )
        return 1
    raise ValueError(
        f"Cannot find K/V split axis in paged KV cache for {layer_name}: {tuple(dst_shape)}"
    )


def match_tail_shape(
    src_k: torch.Tensor, src_v: torch.Tensor, dst_tail: tuple[int, ...], layer_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape ``[tokens, heads, head_dim]`` sources to a cache that flattens the tail."""
    src_tail = tuple(src_k.shape[1:])
    if dst_tail == src_tail:
        return src_k, src_v
    if len(dst_tail) == 1 and len(src_tail) == 2 and dst_tail[0] == src_tail[0] * src_tail[1]:
        return src_k.reshape(src_k.shape[0], -1), src_v.reshape(src_v.shape[0], -1)
    raise ValueError(f"Tail shape mismatch for {layer_name}: dst={dst_tail}, src={src_tail}")
