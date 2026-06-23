from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from .strict_gpu_registry import get_gpu_memory_store, update_namespace_diagnostics

try:
    from vllm.v1.attention.backends.mla.common import MLACommonMetadata
except ImportError:  # pragma: no cover - depends on installed vLLM build
    _MLA_METADATA_TYPES: tuple[type[Any], ...] = ()
else:
    _MLA_METADATA_TYPES = (MLACommonMetadata,)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MemoryLoadMeta:
    user_id: str
    slot_mapping: torch.Tensor
    num_tokens: int


@dataclass
class MemoryConnectorMetadata(KVConnectorMetadata):
    loads: list[MemoryLoadMeta] = field(default_factory=list)

    def add_load(
        self,
        *,
        user_id: str,
        block_ids: list[int],
        block_size: int,
        num_tokens: int,
    ) -> None:
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + block_ids_tensor.reshape(block_ids_tensor.shape[0], 1) * block_size
        )
        self.loads.append(
            MemoryLoadMeta(
                user_id=user_id,
                slot_mapping=slot_mapping.flatten()[:num_tokens],
                num_tokens=num_tokens,
            )
        )


@dataclass(frozen=True)
class PrefixMatch:
    matched: bool
    aligned_tokens: int = 0
    raw_memory_tokens: int = 0
    miss_reason: str = ""
    mismatch_index: int = -1


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens // block_size) * block_size


def _extra_config(kv_transfer_config: Any, key: str, default: Any = None) -> Any:
    getter = getattr(kv_transfer_config, "get_from_extra_config", None)
    if callable(getter):
        return getter(key, default)
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    return extra.get(key, default)


def _extract_user_id(request: "Request", default_user_id: str | None = None) -> str | None:
    user_id = getattr(request, "user", None)
    if user_id:
        return str(user_id)

    user_id = _extract_user_id_from_kv_transfer_params(getattr(request, "kv_transfer_params", None))
    if user_id:
        return user_id

    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is not None:
        user_id = getattr(sampling_params, "user", None)
        if user_id:
            return str(user_id)
        extra_args = getattr(sampling_params, "extra_args", None)
        if isinstance(extra_args, dict):
            user_id = _extract_user_id_from_kv_transfer_params(extra_args.get("kv_transfer_params"))
            if user_id:
                return user_id
            user_id = extra_args.get("user_id") or extra_args.get("user")
            if user_id:
                return str(user_id)

    metadata = getattr(request, "metadata", None)
    if metadata:
        user_id = metadata.get("user_id")
        if user_id:
            return str(user_id)

    if default_user_id:
        return default_user_id
    return None


def _extract_user_id_from_kv_transfer_params(params: Any) -> str | None:
    if not isinstance(params, dict):
        return None
    user_id = params.get("user_id") or params.get("user")
    if user_id:
        return str(user_id)
    return None


def _first_mismatch_index(left: list[int], right: list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return -1


def _is_mla_metadata(value: Any) -> bool:
    return bool(_MLA_METADATA_TYPES) and isinstance(value, _MLA_METADATA_TYPES)


def _store_user_count(store: Any) -> int:
    try:
        return len(store.get_all_user_ids())
    except Exception:
        return 0


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
                "Register GPU tensors through locomo_jasper_bench.kv.strict_gpu_registry."
            )

        namespace = str(_extra_config(self._kv_transfer_config, "memory_namespace", "default"))
        self._strict_memory_namespace = namespace
        self._memory_store = get_gpu_memory_store(namespace)
        self._default_user_id = _extra_config(self._kv_transfer_config, "default_user_id")
        self._allow_prefix_scan = bool(_extra_config(self._kv_transfer_config, "allow_prefix_scan", False))
        self._requests_need_load: dict[str, tuple[str, int]] = {}
        update_namespace_diagnostics(
            namespace,
            increments={"connector_init_count": 1},
            values={
                "connector_block_size": self._block_size,
                "connector_last_role": str(role),
                "connector_store_id": id(self._memory_store),
                "connector_store_user_count": _store_user_count(self._memory_store),
            },
        )

        if self._allow_prefix_scan:
            logger.warning(
                "allow_prefix_scan=True scans all strict GPU memory users for unmatched requests. "
                "This is unsafe for benchmark comparisons unless explicitly intended."
            )
        logger.info(
            "Strict MemoryKVConnector initialized role=%s namespace=%s block_size=%d store_id=%s users=%d",
            role,
            namespace,
            self._block_size,
            id(self._memory_store),
            _store_user_count(self._memory_store),
        )

    @property
    def memory_store(self) -> Any:
        return self._memory_store

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        prompt_token_ids = list(getattr(request, "prompt_token_ids", None) or [])
        request_id = str(getattr(request, "request_id", "") or "")
        user_id = _extract_user_id(request, self._default_user_id)
        base_values = {
            "connector_last_user_id": str(user_id or ""),
            "connector_last_prompt_tokens": len(prompt_token_ids),
            "connector_last_num_computed_tokens": int(num_computed_tokens),
            "connector_last_request_id": request_id,
            "connector_last_miss_reason": "",
            "connector_last_mismatch_index": -1,
            "connector_store_id": id(self._memory_store),
            "connector_store_user_count": _store_user_count(self._memory_store),
        }
        update_namespace_diagnostics(
            self._strict_memory_namespace,
            increments={"connector_match_attempts": 1},
            values=base_values,
        )
        if not prompt_token_ids:
            self._record_match_miss("no_prompt_tokens", user_id=user_id)
            return 0, False

        if user_id is not None:
            match = self._try_match_user(user_id, prompt_token_ids)
        elif self._allow_prefix_scan:
            match = PrefixMatch(matched=False, miss_reason="no_user_id")
            for candidate_user_id in self._memory_store.get_all_user_ids():
                match = self._try_match_user(candidate_user_id, prompt_token_ids)
                if match.matched:
                    user_id = candidate_user_id
                    logger.warning(
                        "Prefix scan matched strict GPU memory user %s for a request with no explicit user id.",
                        candidate_user_id,
                    )
                    break
        else:
            self._record_match_miss("no_user_id", user_id=user_id)
            return 0, False

        if not match.matched or user_id is None:
            self._record_match_miss(
                match.miss_reason or "prefix_mismatch",
                user_id=user_id,
                raw_memory_tokens=match.raw_memory_tokens,
                mismatch_index=match.mismatch_index,
            )
            return 0, False

        aligned_tokens = match.aligned_tokens
        raw_memory_tokens = match.raw_memory_tokens
        new_tokens = aligned_tokens - num_computed_tokens
        if new_tokens <= 0:
            self._record_match_miss(
                "no_new_tokens",
                user_id=user_id,
                aligned_tokens=aligned_tokens,
                raw_memory_tokens=raw_memory_tokens,
                new_tokens=new_tokens,
            )
            return 0, False

        request._memory_user_id = user_id  # type: ignore[attr-defined]
        update_namespace_diagnostics(
            self._strict_memory_namespace,
            increments={"connector_match_hits": 1},
            values={
                "connector_last_user_id": str(user_id),
                "connector_last_raw_memory_tokens": raw_memory_tokens,
                "connector_last_aligned_tokens": aligned_tokens,
                "connector_last_new_tokens": new_tokens,
                "connector_last_miss_reason": "",
                "connector_last_mismatch_index": -1,
                "connector_store_id": id(self._memory_store),
                "connector_store_user_count": _store_user_count(self._memory_store),
            },
        )
        logger.info(
            "Strict GPU memory hit user=%s aligned_tokens=%d raw_tokens=%d new_tokens=%d",
            user_id,
            aligned_tokens,
            raw_memory_tokens,
            new_tokens,
        )
        return new_tokens, False

    def _try_match_user(self, user_id: str, prompt_token_ids: list[int]) -> PrefixMatch:
        memory = self._memory_store.get_user_memory(user_id)
        if memory is None:
            return PrefixMatch(matched=False, miss_reason="memory_missing")
        if memory.token_ids is None:
            return PrefixMatch(matched=False, miss_reason="token_ids_missing")

        memory_token_ids = list(memory.token_ids)
        num_memory_tokens = len(memory_token_ids)
        if len(prompt_token_ids) <= num_memory_tokens:
            return PrefixMatch(
                matched=False,
                raw_memory_tokens=num_memory_tokens,
                miss_reason="prompt_too_short",
            )
        if prompt_token_ids[:num_memory_tokens] != memory_token_ids:
            return PrefixMatch(
                matched=False,
                raw_memory_tokens=num_memory_tokens,
                miss_reason="prefix_mismatch",
                mismatch_index=_first_mismatch_index(
                    prompt_token_ids[:num_memory_tokens],
                    memory_token_ids,
                ),
            )

        aligned_tokens = align_to_block_size(num_memory_tokens, self._block_size)
        if aligned_tokens <= 0:
            return PrefixMatch(
                matched=False,
                raw_memory_tokens=num_memory_tokens,
                miss_reason="aligned_zero",
            )
        return PrefixMatch(
            matched=True,
            aligned_tokens=aligned_tokens,
            raw_memory_tokens=num_memory_tokens,
        )

    def _record_match_miss(
        self,
        reason: str,
        *,
        user_id: str | None,
        aligned_tokens: int = 0,
        raw_memory_tokens: int = 0,
        new_tokens: int = 0,
        mismatch_index: int = -1,
    ) -> None:
        update_namespace_diagnostics(
            self._strict_memory_namespace,
            increments={"connector_match_misses": 1},
            values={
                "connector_last_user_id": str(user_id or ""),
                "connector_last_miss_reason": reason,
                "connector_last_aligned_tokens": int(aligned_tokens),
                "connector_last_raw_memory_tokens": int(raw_memory_tokens),
                "connector_last_new_tokens": int(new_tokens),
                "connector_last_mismatch_index": int(mismatch_index),
                "connector_store_id": id(self._memory_store),
                "connector_store_user_count": _store_user_count(self._memory_store),
            },
        )

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        del blocks
        update_namespace_diagnostics(
            self._strict_memory_namespace,
            increments={"connector_update_state_calls": 1},
            values={
                "connector_last_request_id": str(getattr(request, "request_id", "") or ""),
                "connector_last_new_tokens": int(num_external_tokens),
            },
        )
        if num_external_tokens <= 0:
            return

        user_id = getattr(request, "_memory_user_id", None)
        if user_id is None:
            user_id = _extract_user_id(request, self._default_user_id)
        if user_id is None:
            return

        self._requests_need_load[request.request_id] = (str(user_id), int(num_external_tokens))
        logger.debug(
            "Scheduled strict GPU memory load req=%s user=%s tokens=%d",
            request.request_id,
            user_id,
            num_external_tokens,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        update_namespace_diagnostics(
            self._strict_memory_namespace,
            increments={"connector_build_meta_calls": 1},
        )
        meta = MemoryConnectorMetadata()
        handled_req_ids: list[str] = []

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            if req_id not in self._requests_need_load:
                continue
            user_id, num_memory_tokens = self._requests_need_load[req_id]
            num_memory_blocks = (num_memory_tokens + self._block_size - 1) // self._block_size
            memory_block_ids = new_req.block_ids[0][:num_memory_blocks]
            meta.add_load(
                user_id=user_id,
                block_ids=memory_block_ids,
                block_size=self._block_size,
                num_tokens=num_memory_tokens,
            )
            handled_req_ids.append(req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached_reqs.req_ids):
            if not cached_reqs.resumed_from_preemption[index] or req_id not in self._requests_need_load:
                continue

            new_block_ids = cached_reqs.new_block_ids[index]
            if new_block_ids is None:
                continue

            user_id, num_memory_tokens = self._requests_need_load[req_id]
            num_memory_blocks = (num_memory_tokens + self._block_size - 1) // self._block_size
            memory_block_ids = new_block_ids[0][:num_memory_blocks]
            meta.add_load(
                user_id=user_id,
                block_ids=memory_block_ids,
                block_size=self._block_size,
                num_tokens=num_memory_tokens,
            )
            handled_req_ids.append(req_id)

        for req_id in handled_req_ids:
            self._requests_need_load.pop(req_id, None)

        if meta.loads:
            update_namespace_diagnostics(
                self._strict_memory_namespace,
                increments={"connector_metadata_loads": len(meta.loads)},
            )
            logger.info("Built strict GPU connector metadata for %d memory loads", len(meta.loads))
        return meta

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del kwargs
        update_namespace_diagnostics(
            self._strict_memory_namespace,
            increments={"connector_start_load_calls": 1},
        )
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, MemoryConnectorMetadata):
            raise TypeError(f"Unexpected connector metadata type: {type(metadata)!r}")

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("start_load_kv called with no attention metadata.")
            return

        for load in metadata.loads:
            memory = self._memory_store.get_user_memory(load.user_id)
            if memory is None:
                update_namespace_diagnostics(
                    self._strict_memory_namespace,
                    increments={"connector_missing_memory_loads": 1},
                )
                logger.warning("Strict GPU memory for user %s was not found during load.", load.user_id)
                continue

            first_tensor = next(iter(memory.kv_by_layer.values()), None)
            if first_tensor is None:
                update_namespace_diagnostics(
                    self._strict_memory_namespace,
                    increments={"connector_missing_memory_loads": 1},
                )
                logger.warning("Strict GPU memory for user %s has no layer tensors.", load.user_id)
                continue

            slot_mapping = load.slot_mapping.to(device=first_tensor.device, dtype=torch.long)
            update_namespace_diagnostics(
                self._strict_memory_namespace,
                increments={"connector_injected_tokens": load.num_tokens},
                values={
                    "connector_last_user_id": load.user_id,
                    "connector_last_new_tokens": load.num_tokens,
                    "connector_store_id": id(self._memory_store),
                    "connector_store_user_count": _store_user_count(self._memory_store),
                },
            )
            logger.info("Injecting %d strict GPU memory tokens for user %s", load.num_tokens, load.user_id)

            for layer_name, layer in forward_context.no_compile_layers.items():
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue

                src_kv = memory.kv_by_layer.get(layer_name)
                if src_kv is None:
                    update_namespace_diagnostics(
                        self._strict_memory_namespace,
                        increments={"connector_missing_layer_loads": 1},
                    )
                    logger.warning("Layer %s not found in strict GPU memory for user %s", layer_name, load.user_id)
                    continue

                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                self._inject_kv_into_layer(
                    dst_kv_cache_layer=kv_cache_layer,
                    src_kv_cache=self._truncate_kv(src_kv, load.num_tokens),
                    slot_mapping=slot_mapping,
                    attn_metadata=attn_metadata,
                    layer_name=layer_name,
                )

    @staticmethod
    def _truncate_kv(src_kv: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if src_kv.ndim >= 4 and src_kv.shape[0] == 2:
            return src_kv[:, :num_tokens]
        return src_kv[:num_tokens]

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
