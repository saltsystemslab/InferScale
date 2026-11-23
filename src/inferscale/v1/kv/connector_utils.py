from __future__ import annotations

from typing import Any

MEMORY_USER_ID_EXTRA_ARG = "inferscale_memory_user_id"

DEFAULT_KV_STORE_BACKEND = "gpu"
KNOWN_KV_STORE_BACKENDS = ("gpu", "cpu")
DEFAULT_KV_STAGING_SLOTS = 4


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens // block_size) * block_size


def build_slot_mapping(block_ids: list[int], block_size: int, num_tokens: int) -> list[int]:
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0.")
    if num_tokens > len(block_ids) * block_size:
        raise ValueError(
            f"num_tokens={num_tokens} exceeds the allocated capacity "
            f"{len(block_ids)} blocks x {block_size}."
        )
    slots: list[int] = []
    for block_id in block_ids:
        first_slot = block_id * block_size
        slots.extend(range(first_slot, first_slot + block_size))
    return slots[:num_tokens]


def extra_config(kv_transfer_config: Any, key: str, default: Any = None) -> Any:
    getter = getattr(kv_transfer_config, "get_from_extra_config", None)
    if callable(getter):
        return getter(key, default)
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    return extra.get(key, default)


def extract_user_id(request: Any, default_user_id: str | None = None) -> str | None:
    request_user = getattr(request, "user", None)
    if request_user:
        return str(request_user)
    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is not None:
        sampling_user = getattr(sampling_params, "user", None)
        if sampling_user:
            return str(sampling_user)
        extra_args = getattr(sampling_params, "extra_args", None)
        if isinstance(extra_args, dict):
            memory_user = extra_args.get(MEMORY_USER_ID_EXTRA_ARG)
            if memory_user:
                return str(memory_user)
    metadata = getattr(request, "metadata", None)
    if metadata:
        metadata_user = metadata.get("user_id")
        if metadata_user:
            return str(metadata_user)
    return default_user_id or None
