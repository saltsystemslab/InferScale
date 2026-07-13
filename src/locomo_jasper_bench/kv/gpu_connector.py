from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import torch

from vllm.v1.attention.backend import AttentionMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from .connector_metadata import (
    MemoryConnectorMetadata,
    align_to_block_size,
    extra_config as _extra_config,
    extract_user_id as _extract_user_id,
)
from .gpu_registry import get_gpu_memory_store

from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata

_MLA_METADATA_TYPES = (MLACommonMetadata,)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Strict GPU mode runs the vLLM engine in-process, so the benchmark worker can
# read these counters to verify that every measured request actually had its
# KV loaded (a load that never happens would otherwise be invisible).
_LOAD_STATS_LOCK = threading.Lock()
_LOAD_STATS = {"requests_loaded": 0, "tokens_loaded": 0}
# Request ids that had memory injected, and request ids whose memory region
# was fully covered by the native prefix cache (a legitimate no-load). Their
# union is the set of requests that generated WITH their memory available,
# which is what the benchmark's fail-fast check needs to be exact.
_LOADED_REQUEST_IDS: set[str] = set()
_NATIVE_COVERED_REQUEST_IDS: set[str] = set()


def reset_load_stats() -> None:
    with _LOAD_STATS_LOCK:
        _LOAD_STATS["requests_loaded"] = 0
        _LOAD_STATS["tokens_loaded"] = 0
        _LOADED_REQUEST_IDS.clear()
        _NATIVE_COVERED_REQUEST_IDS.clear()


def snapshot_load_stats() -> dict[str, int]:
    with _LOAD_STATS_LOCK:
        return {
            **_LOAD_STATS,
            "requests_covered": len(_LOADED_REQUEST_IDS | _NATIVE_COVERED_REQUEST_IDS),
        }


def _record_load(num_tokens: int, request_id: str = "") -> None:
    with _LOAD_STATS_LOCK:
        _LOAD_STATS["requests_loaded"] += 1
        _LOAD_STATS["tokens_loaded"] += int(num_tokens)
        if request_id:
            _LOADED_REQUEST_IDS.add(str(request_id))


def _record_native_covered(request_id: str) -> None:
    with _LOAD_STATS_LOCK:
        if request_id:
            _NATIVE_COVERED_REQUEST_IDS.add(str(request_id))


def _is_mla_metadata(value: Any) -> bool:
    return bool(_MLA_METADATA_TYPES) and isinstance(value, _MLA_METADATA_TYPES)


class MemoryKVConnector(KVConnectorBase_V1):
    """Strict GPU-only KV connector for benchmark-composed memory tensors."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Any | None = None,
    ) -> None:
        del kv_cache_config
        super().__init__(vllm_config=vllm_config, role=role)
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._block_size = int(vllm_config.cache_config.block_size)

        memory_path = _extra_config(self._kv_transfer_config, "memory_path")
        if memory_path is not None:
            raise RuntimeError(
                "Strict GPU KV mode does not allow memory_path or disk-backed KV loading. "
                "Register GPU tensors through locomo_jasper_bench.kv.gpu_registry."
        )

        namespace = str(_extra_config(self._kv_transfer_config, "memory_namespace", "default"))
        store_backend = str(_extra_config(self._kv_transfer_config, "memory_store_backend", "gpu"))
        num_staging_slots = int(_extra_config(self._kv_transfer_config, "num_staging_slots", 4))
        self._memory_store = get_gpu_memory_store(
            namespace,
            backend=store_backend,
            num_staging_slots=num_staging_slots,
        )
        self._default_user_id = _extra_config(self._kv_transfer_config, "default_user_id")
        self._allow_prefix_scan = bool(_extra_config(self._kv_transfer_config, "allow_prefix_scan", False))
        self._log_memory_hits = bool(_extra_config(self._kv_transfer_config, "log_memory_hits", True))
        self._requests_need_load: dict[str, tuple[str, int, int]] = {}

        if self._allow_prefix_scan:
            logger.warning(
                "allow_prefix_scan=True scans all strict GPU memory users for unmatched requests. "
                "This is unsafe for benchmark comparisons unless explicitly intended."
            )
        logger.info(
            "Strict MemoryKVConnector initialized role=%s namespace=%s block_size=%d",
            role,
            namespace,
            self._block_size,
        )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        prompt_token_ids = list(getattr(request, "prompt_token_ids", None) or [])
        if not prompt_token_ids:
            return 0, False

        user_id = _extract_user_id(request, self._default_user_id)
        if user_id is not None:
            match = self._try_match_user(user_id, prompt_token_ids)
            if match is None:
                raise RuntimeError(
                    "Explicit strict GPU memory routing failed: "
                    f"user_id={user_id!r} request_id={request.request_id!r}."
                )
        elif self._allow_prefix_scan:
            match = None
            for candidate_user_id in self._memory_store.get_all_user_ids():
                match = self._try_match_user(candidate_user_id, prompt_token_ids)
                if match is not None:
                    user_id = candidate_user_id
                    log_match = logger.warning if self._log_memory_hits else logger.debug
                    log_match(
                        "Prefix scan matched strict GPU memory user %s for a request with no explicit user id.",
                        candidate_user_id,
                    )
                    break
        else:
            return 0, False

        if match is None or user_id is None:
            return 0, False

        aligned_tokens, raw_memory_tokens = match
        new_tokens = aligned_tokens - num_computed_tokens
        if new_tokens <= 0:
            # The native prefix cache fully covers this request's memory
            # region: a legitimate no-load that still counts as covered.
            _record_native_covered(str(request.request_id))
            return 0, False

        request._memory_user_id = user_id  # type: ignore[attr-defined]
        # With prefix caching enabled, num_computed_tokens can cover a
        # block-aligned front of the memory region; the load must then skip
        # those tokens and target the blocks after the cached ones.
        request._memory_skip_tokens = int(num_computed_tokens)  # type: ignore[attr-defined]
        log_hit = logger.info if self._log_memory_hits else logger.debug
        log_hit(
            "Strict GPU memory hit user=%s aligned_tokens=%d raw_tokens=%d new_tokens=%d",
            user_id,
            aligned_tokens,
            raw_memory_tokens,
            new_tokens,
        )
        return new_tokens, False

    def _try_match_user(self, user_id: str, prompt_token_ids: list[int]) -> tuple[int, int] | None:
        # Metadata-only read: matching must never stage KV (the cpu-pinned
        # store's get_user_memory issues a full H2D prefetch).
        memory = self._memory_store.peek_user_memory(user_id)
        if memory is None or memory.token_ids is None:
            return None

        memory_token_ids = list(memory.token_ids)
        num_memory_tokens = len(memory_token_ids)
        if len(prompt_token_ids) <= num_memory_tokens:
            return None
        if prompt_token_ids[:num_memory_tokens] != memory_token_ids:
            return None

        aligned_tokens = align_to_block_size(num_memory_tokens, self._block_size)
        if aligned_tokens <= 0:
            return None
        return aligned_tokens, num_memory_tokens

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        del blocks
        if num_external_tokens <= 0:
            return

        user_id = getattr(request, "_memory_user_id", None)
        if user_id is None:
            user_id = _extract_user_id(request, self._default_user_id)
        if user_id is None:
            return

        skip_tokens = int(getattr(request, "_memory_skip_tokens", 0))
        self._requests_need_load[request.request_id] = (
            str(user_id),
            int(num_external_tokens),
            skip_tokens,
        )
        logger.debug(
            "Scheduled strict GPU memory load req=%s user=%s tokens=%d skip=%d",
            request.request_id,
            user_id,
            num_external_tokens,
            skip_tokens,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = MemoryConnectorMetadata()
        handled_req_ids: list[str] = []

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            if req_id not in self._requests_need_load:
                continue
            user_id, num_memory_tokens, skip_tokens = self._requests_need_load[req_id]
            if skip_tokens % self._block_size != 0:
                raise RuntimeError(
                    f"Native prefix-cache hit of {skip_tokens} tokens for req {req_id} "
                    f"is not block-aligned (block_size={self._block_size})."
                )
            skip_blocks = skip_tokens // self._block_size
            num_memory_blocks = (num_memory_tokens + self._block_size - 1) // self._block_size
            memory_block_ids = new_req.block_ids[0][skip_blocks : skip_blocks + num_memory_blocks]
            if len(memory_block_ids) != num_memory_blocks:
                # Fires if vLLM's NewRequestData.block_ids does not include the
                # natively cached front blocks; the offset assumption must then
                # be revisited rather than injecting into the wrong blocks.
                raise RuntimeError(
                    f"Request {req_id} exposes {len(new_req.block_ids[0])} block(s); "
                    f"expected at least {skip_blocks + num_memory_blocks} "
                    f"(skip={skip_blocks}, load={num_memory_blocks})."
                )
            if skip_tokens > 0:
                logger.info(
                    "Partial native prefix hit: skipping %d cached tokens (%d blocks) "
                    "before injecting %d memory tokens for user %s",
                    skip_tokens,
                    skip_blocks,
                    num_memory_tokens,
                    user_id,
                )
            meta.add_load(
                user_id=user_id,
                block_ids=memory_block_ids,
                block_size=self._block_size,
                num_tokens=num_memory_tokens,
                skip_tokens=skip_tokens,
                request_id=str(req_id),
            )
            handled_req_ids.append(req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        resumed_req_ids = set(cached_reqs.resumed_req_ids)
        for index, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in resumed_req_ids or req_id not in self._requests_need_load:
                continue

            new_block_ids = cached_reqs.new_block_ids[index]
            if new_block_ids is None:
                continue

            user_id, num_memory_tokens, skip_tokens = self._requests_need_load[req_id]
            if skip_tokens > 0:
                # A resumed request whose memory region was partially covered by
                # the native prefix cache has no verified block layout; fail
                # closed instead of guessing where the tail blocks landed.
                raise RuntimeError(
                    f"Resumed request {req_id} has a partial native prefix hit "
                    f"({skip_tokens} tokens); this combination is unsupported."
                )
            num_memory_blocks = (num_memory_tokens + self._block_size - 1) // self._block_size
            memory_block_ids = new_block_ids[0][:num_memory_blocks]
            meta.add_load(
                user_id=user_id,
                block_ids=memory_block_ids,
                block_size=self._block_size,
                num_tokens=num_memory_tokens,
                request_id=str(req_id),
            )
            handled_req_ids.append(req_id)

        for req_id in handled_req_ids:
            self._requests_need_load.pop(req_id, None)

        if meta.loads:
            logger.info("Built strict GPU connector metadata for %d memory loads", len(meta.loads))
        return meta

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, MemoryConnectorMetadata):
            raise TypeError(f"Unexpected connector metadata type: {type(metadata)!r}")
        if not metadata.loads:
            # Decode-only steps must stay zero-cost: no staging traffic and
            # no store interaction of any kind.
            return

        # By the time start_load_kv runs, the scheduler has already credited
        # the memory tokens as externally computed and skipped their prefill,
        # so any failure to load below must fail closed: continuing would let
        # the request attend over uninitialized KV.
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            raise RuntimeError(
                "start_load_kv called with no attention metadata while "
                f"{len(metadata.loads)} strict GPU memory load(s) are pending."
            )

        # Pinned-host store: slide a prefetch window sized to the staging
        # pool. Staging every load up front would evict just-prefetched
        # users whenever one step carries more loads than slots; instead
        # each release below frees the slot the next pending load fills.
        # No-ops for the GPU store.
        loads = metadata.loads
        prefetch = getattr(self._memory_store, "prefetch_user_to_gpu", None)
        release = getattr(self._memory_store, "release_staging", None)
        window = len(loads)
        if callable(prefetch):
            slots = int(getattr(self._memory_store, "num_staging_slots", 0) or 0)
            if slots > 0:
                window = min(slots, len(loads))
            for load in loads[:window]:
                prefetch(load.user_id)

        try:
            for index, load in enumerate(loads):
                self._inject_one_load(load, forward_context, attn_metadata)
                if callable(release):
                    release(load.user_id)
                if callable(prefetch) and index + window < len(loads):
                    prefetch(loads[index + window].user_id)
        finally:
            # release_staging is idempotent; sweeping every load covers the
            # error path without double-recording the successful ones.
            if callable(release):
                for load in loads:
                    release(load.user_id)

    def _inject_one_load(
        self,
        load: "MemoryLoadMeta",
        forward_context: "ForwardContext",
        attn_metadata: AttentionMetadata,
    ) -> None:
        memory = self._memory_store.get_user_memory(load.user_id)
        if memory is None:
            raise RuntimeError(
                f"Strict GPU memory for user {load.user_id} was not found during load."
            )

        # Probe one layer for the device; the pinned-host view waits on a
        # layer's copy event on first access, so touch exactly one.
        layer_names = list(memory.kv_by_layer.keys())
        first_tensor = memory.kv_by_layer[layer_names[0]] if layer_names else None
        if first_tensor is None:
            raise RuntimeError(
                f"Strict GPU memory for user {load.user_id} has no layer tensors."
            )

        slot_mapping = load.slot_mapping.to(device=first_tensor.device, dtype=torch.long)
        log_injection = logger.info if self._log_memory_hits else logger.debug
        log_injection("Injecting %d strict GPU memory tokens for user %s", load.num_tokens, load.user_id)

        injected_layers = 0
        for layer_name, layer in forward_context.no_compile_layers.items():
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue

            src_kv = memory.kv_by_layer.get(layer_name)
            if src_kv is None:
                raise RuntimeError(
                    f"Layer {layer_name} not found in strict GPU memory for "
                    f"user {load.user_id}."
                )

            self._inject_kv_into_layer(
                dst_kv_cache_layer=kv_cache_attr,
                src_kv_cache=self._slice_kv(src_kv, load.skip_tokens, load.num_tokens),
                slot_mapping=slot_mapping,
                attn_metadata=attn_metadata,
                layer_name=layer_name,
            )
            injected_layers += 1

        if injected_layers == 0:
            raise RuntimeError(
                f"No KV cache layers were injected for user {load.user_id}."
            )
        _record_load(load.num_tokens, load.request_id)

    @staticmethod
    def _slice_kv(src_kv: torch.Tensor, skip_tokens: int, num_tokens: int) -> torch.Tensor:
        if src_kv.ndim >= 4 and src_kv.shape[0] == 2:
            return src_kv[:, skip_tokens : skip_tokens + num_tokens]
        return src_kv[skip_tokens : skip_tokens + num_tokens]

    def _inject_kv_into_layer(
        self,
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

        if _is_mla_metadata(layer_meta):
            pages = dst_kv_cache_layer.shape[0]
            page_size = dst_kv_cache_layer.shape[1]
            dst_flat = dst_kv_cache_layer.reshape(pages * page_size, -1)
            dst_flat[slot_mapping] = src_kv_cache.to(
                dtype=dst_kv_cache_layer.dtype,
                device=dst_kv_cache_layer.device,
            )
            return

        if isinstance(layer_meta, TritonAttentionMetadata):
            logger.debug("Injecting strict GPU KV for Triton attention layer %s", layer_name)

        if src_kv_cache.ndim < 4 or src_kv_cache.shape[0] != 2:
            raise ValueError(
                f"Expected source KV for {layer_name} to have shape [2, tokens, ...], "
                f"got {tuple(src_kv_cache.shape)}"
            )

        dst_shape = dst_kv_cache_layer.shape
        if dst_kv_cache_layer.ndim < 4:
            raise ValueError(f"Unexpected paged KV cache shape for {layer_name}: {tuple(dst_shape)}")

        shape_0, shape_1 = dst_shape[0], dst_shape[1]
        if shape_1 == 2 and shape_0 != 2:
            kv_split_dim = 1
        elif shape_0 == 2 and shape_1 != 2:
            kv_split_dim = 0
        elif shape_0 == 2 and shape_1 == 2:
            kv_split_dim = 1
            if not getattr(self, "_warned_ambiguous_layout", False):
                logger.warning(
                    "Ambiguous strict GPU KV cache layout for %s shape=%s; defaulting to split dim 1.",
                    layer_name,
                    tuple(dst_shape),
                )
                self._warned_ambiguous_layout = True
        else:
            raise ValueError(
                f"Cannot find K/V split axis in paged KV cache for {layer_name}: {tuple(dst_shape)}"
            )

        page_size = dst_shape[2]
        slots = slot_mapping.to(device=dst_kv_cache_layer.device, dtype=torch.long)
        pages = slots // page_size
        offsets = slots % page_size

        src_k = src_kv_cache[0].to(dtype=dst_kv_cache_layer.dtype, device=dst_kv_cache_layer.device)
        src_v = src_kv_cache[1].to(dtype=dst_kv_cache_layer.dtype, device=dst_kv_cache_layer.device)
        src_tail = tuple(src_k.shape[1:])
        dst_tail = tuple(dst_shape[3:])

        if dst_tail == src_tail:
            src_k_reshaped = src_k
            src_v_reshaped = src_v
        elif len(dst_tail) == 1 and len(src_tail) == 2 and dst_tail[0] == src_tail[0] * src_tail[1]:
            src_k_reshaped = src_k.reshape(src_k.shape[0], -1)
            src_v_reshaped = src_v.reshape(src_v.shape[0], -1)
        else:
            raise ValueError(
                f"Tail shape mismatch for {layer_name}: dst={dst_tail}, src={src_tail}"
            )

        if kv_split_dim == 0:
            dst_kv_cache_layer[0, pages, offsets] = src_k_reshaped
            dst_kv_cache_layer[1, pages, offsets] = src_v_reshaped
        else:
            dst_kv_cache_layer[pages, 0, offsets] = src_k_reshaped
            dst_kv_cache_layer[pages, 1, offsets] = src_v_reshaped

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs
        return

    def wait_for_save(self) -> None:
        return

    def shutdown(self) -> None:
        stats = self._memory_store.get_stats()
        logger.info(
            "Strict MemoryKVConnector shutting down: users=%d tokens=%d gpu_mb=%.1f",
            stats.get("num_users", 0),
            stats.get("total_tokens", 0),
            stats.get("total_gpu_mb", 0.0),
        )


__all__ = ["MemoryKVConnector"]
