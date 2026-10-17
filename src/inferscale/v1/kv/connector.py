"""vLLM KV connector that injects composed memory KV into scheduled requests.

Scheduler side: a request whose prompt starts with a registered memory's
token ids is credited with those tokens as externally computed, so vLLM
skips their prefill. Worker side: before the forward pass, the memory's KV
is scattered into the request's allocated blocks. Memories reach the
connector through the namespace registry in this process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from .connector_inject import inject_load
from .connector_metadata import MemoryConnectorMetadata, MemoryLoadMeta
from .connector_utils import (
    DEFAULT_KV_STAGING_SLOTS,
    DEFAULT_KV_STORE_BACKEND,
    align_to_block_size,
    extra_config,
    extract_user_id,
)
from .load_stats import LoadStats
from .registry import get_gpu_memory_store

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

load_stats = LoadStats()


def reset_load_stats() -> None:
    load_stats.reset()


def snapshot_load_stats() -> dict[str, int]:
    return load_stats.snapshot()


@dataclass(slots=True, frozen=True)
class _PendingMatch:
    """A matched request that is waiting for its block allocation."""

    user_id: str
    # Memory tokens the native prefix cache already covers; the load skips them.
    skip_tokens: int


@dataclass(slots=True, frozen=True)
class _ScheduledLoad:
    user_id: str
    num_tokens: int
    skip_tokens: int


class MemoryKVConnector(KVConnectorBase_V1):
    """KV connector for composed memory tensors registered in this process."""

    def __init__(
        self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: Any | None = None
    ) -> None:
        del kv_cache_config
        super().__init__(vllm_config=vllm_config, role=role)
        transfer_config = vllm_config.kv_transfer_config
        self._block_size = int(vllm_config.cache_config.block_size)
        namespace = str(extra_config(transfer_config, "memory_namespace", "default"))
        self._memory_store = get_gpu_memory_store(
            namespace,
            backend=str(
                extra_config(transfer_config, "memory_store_backend", DEFAULT_KV_STORE_BACKEND)
            ),
            num_staging_slots=int(
                extra_config(transfer_config, "num_staging_slots", DEFAULT_KV_STAGING_SLOTS)
            ),
        )
        self._default_user_id = extra_config(transfer_config, "default_user_id")
        self._log_memory_hits = bool(extra_config(transfer_config, "log_memory_hits", True))
        self._pending_matches: dict[str, _PendingMatch] = {}
        self._scheduled_loads: dict[str, _ScheduledLoad] = {}
        logger.info(
            "MemoryKVConnector initialized role=%s namespace=%s block_size=%d",
            role,
            namespace,
            self._block_size,
        )

    # Scheduler side

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        prompt_token_ids = list(getattr(request, "prompt_token_ids", None) or [])
        if not prompt_token_ids:
            return 0, False
        user_id = extract_user_id(request, self._default_user_id)
        if user_id is None:
            return 0, False
        match = self._try_match_user(user_id, prompt_token_ids)
        if match is None:
            raise RuntimeError(
                f"Memory routing failed: user_id={user_id!r} request_id={request.request_id!r}."
            )
        aligned_tokens, raw_memory_tokens = match
        new_tokens = aligned_tokens - num_computed_tokens
        if new_tokens <= 0:
            # The native prefix cache fully covers this request's memory
            # region: a legitimate no-load that still counts as covered.
            load_stats.record_native_covered(str(request.request_id))
            return 0, False
        # With prefix caching enabled, num_computed_tokens can cover a
        # block-aligned front of the memory region; the load then skips those
        # tokens and targets the blocks after the cached ones.
        self._pending_matches[request.request_id] = _PendingMatch(user_id, int(num_computed_tokens))
        self._log_hit(
            "Memory hit user=%s aligned_tokens=%d raw_tokens=%d new_tokens=%d",
            user_id,
            aligned_tokens,
            raw_memory_tokens,
            new_tokens,
        )
        return new_tokens, False

    def _try_match_user(self, user_id: str, prompt_token_ids: list[int]) -> tuple[int, int] | None:
        # Metadata-only read: matching must never stage KV (the cpu store's
        # get_user_memory issues a full H2D prefetch).
        memory = self._memory_store.peek_user_memory(user_id)
        if memory is None or memory.token_ids is None:
            return None
        num_memory_tokens = len(memory.token_ids)
        if len(prompt_token_ids) <= num_memory_tokens:
            return None
        if prompt_token_ids[:num_memory_tokens] != memory.token_ids:
            return None
        aligned_tokens = align_to_block_size(num_memory_tokens, self._block_size)
        if aligned_tokens <= 0:
            return None
        return aligned_tokens, num_memory_tokens

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        del blocks
        pending = self._pending_matches.pop(request.request_id, None)
        if num_external_tokens <= 0:
            return
        if pending is not None:
            user_id, skip_tokens = pending.user_id, pending.skip_tokens
        else:
            user_id = extract_user_id(request, self._default_user_id)
            skip_tokens = 0
        if user_id is None:
            return
        self._scheduled_loads[request.request_id] = _ScheduledLoad(
            str(user_id), int(num_external_tokens), skip_tokens
        )
        logger.debug(
            "Scheduled memory load req=%s user=%s tokens=%d skip=%d",
            request.request_id,
            user_id,
            num_external_tokens,
            skip_tokens,
        )

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Forget a match whose request ended before allocation; blocks free synchronously."""
        del block_ids
        self._pending_matches.pop(request.request_id, None)
        return False, None

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = MemoryConnectorMetadata()
        handled: list[str] = []
        for new_req in scheduler_output.scheduled_new_reqs:
            load = self._scheduled_loads.get(new_req.req_id)
            if load is None:
                continue
            self._add_new_request_load(meta, new_req.req_id, new_req.block_ids[0], load)
            handled.append(new_req.req_id)
        cached = scheduler_output.scheduled_cached_reqs
        resumed = set(cached.resumed_req_ids)
        for index, req_id in enumerate(cached.req_ids):
            load = self._scheduled_loads.get(req_id)
            new_block_ids = cached.new_block_ids[index]
            if load is None or req_id not in resumed or new_block_ids is None:
                continue
            self._add_resumed_request_load(meta, req_id, new_block_ids[0], load)
            handled.append(req_id)
        for req_id in handled:
            self._scheduled_loads.pop(req_id, None)
        if meta.loads:
            logger.info("Built connector metadata for %d memory loads", len(meta.loads))
        return meta

    def _add_new_request_load(
        self, meta: MemoryConnectorMetadata, req_id: str, block_ids: list[int], load: _ScheduledLoad
    ) -> None:
        if load.skip_tokens % self._block_size != 0:
            raise RuntimeError(
                f"Native prefix-cache hit of {load.skip_tokens} tokens for req {req_id} "
                f"is not block-aligned (block_size={self._block_size})."
            )
        skip_blocks = load.skip_tokens // self._block_size
        num_memory_blocks = self._blocks_for(load.num_tokens)
        memory_block_ids = block_ids[skip_blocks : skip_blocks + num_memory_blocks]
        if len(memory_block_ids) != num_memory_blocks:
            # vLLM's NewRequestData.block_ids must include the natively cached
            # front blocks for the offset above to hold; fail rather than
            # inject into the wrong blocks.
            raise RuntimeError(
                f"Request {req_id} exposes {len(block_ids)} block(s); "
                f"expected at least {skip_blocks + num_memory_blocks} "
                f"(skip={skip_blocks}, load={num_memory_blocks})."
            )
        if load.skip_tokens > 0:
            logger.info(
                "Partial native prefix hit: skipping %d cached tokens (%d blocks) "
                "before injecting %d memory tokens for user %s",
                load.skip_tokens,
                skip_blocks,
                load.num_tokens,
                load.user_id,
            )
        meta.add_load(
            user_id=load.user_id,
            block_ids=memory_block_ids,
            block_size=self._block_size,
            num_tokens=load.num_tokens,
            skip_tokens=load.skip_tokens,
            request_id=str(req_id),
        )

    def _add_resumed_request_load(
        self, meta: MemoryConnectorMetadata, req_id: str, block_ids: list[int], load: _ScheduledLoad
    ) -> None:
        if load.skip_tokens > 0:
            # A resumed request whose memory region was partially covered by
            # the native prefix cache has no verified block layout; fail
            # closed instead of guessing where the tail blocks landed.
            raise RuntimeError(
                f"Resumed request {req_id} has a partial native prefix hit "
                f"({load.skip_tokens} tokens); this combination is unsupported."
            )
        meta.add_load(
            user_id=load.user_id,
            block_ids=block_ids[: self._blocks_for(load.num_tokens)],
            block_size=self._block_size,
            num_tokens=load.num_tokens,
            request_id=str(req_id),
        )

    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self._block_size - 1) // self._block_size

    # Worker side

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, MemoryConnectorMetadata):
            raise TypeError(f"Unexpected connector metadata type: {type(metadata)!r}")
        loads = metadata.loads
        if not loads:
            # Decode-only steps must stay zero-cost: no staging traffic and
            # no store interaction of any kind.
            return
        # By now the scheduler has credited the memory tokens as externally
        # computed and skipped their prefill, so any failure below must fail
        # closed: continuing would let the request attend over uninitialized KV.
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            raise RuntimeError(
                "start_load_kv called with no attention metadata while "
                f"{len(loads)} memory load(s) are pending."
            )
        # Pinned-host store: slide a prefetch window sized to the staging
        # pool. Staging every load up front would evict just-prefetched
        # memories whenever one step carries more loads than slots; instead
        # each release frees the slot the next pending load fills. The GPU
        # store reports zero slots, which selects the full window.
        slots = int(self._memory_store.num_staging_slots or 0)
        window = min(slots, len(loads)) if slots > 0 else len(loads)
        for load in loads[:window]:
            self._memory_store.prefetch_user_to_gpu(load.user_id)
        released = 0
        try:
            for index, load in enumerate(loads):
                self._inject_one_load(load, forward_context, attn_metadata)
                self._memory_store.release_staging(load.user_id)
                released = index + 1
                if index + window < len(loads):
                    self._memory_store.prefetch_user_to_gpu(loads[index + window].user_id)
        finally:
            # Error path only: free the slots of loads that never released.
            for load in loads[released:]:
                self._memory_store.release_staging(load.user_id)

    def _inject_one_load(
        self,
        load: MemoryLoadMeta,
        forward_context: ForwardContext,
        attn_metadata: AttentionMetadata,
    ) -> None:
        memory = self._memory_store.get_user_memory(load.user_id)
        if memory is None:
            raise RuntimeError(f"Memory for user {load.user_id} was not found during load.")
        self._log_hit("Injecting %d memory tokens for user %s", load.num_tokens, load.user_id)
        inject_load(memory, load, forward_context, attn_metadata)
        load_stats.record_load(load.num_tokens, load.request_id)

    def _log_hit(self, message: str, *args: Any) -> None:
        if self._log_memory_hits:
            logger.info(message, *args)
        else:
            logger.debug(message, *args)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        return

    def shutdown(self) -> None:
        stats = self._memory_store.get_stats()
        logger.info(
            "MemoryKVConnector shutting down: users=%d tokens=%d gpu_mb=%.1f",
            stats.get("num_users", 0),
            stats.get("total_tokens", 0),
            stats.get("total_gpu_mb", 0.0),
        )


__all__ = ["MemoryKVConnector", "reset_load_stats", "snapshot_load_stats"]
