"""Pure-python connector helpers, kept import-safe without torch or vllm.

connector_metadata.py wraps these for the GPU connector; tests exercise them
directly since the vllm import chain is unavailable off the GPU box.
"""

from __future__ import annotations

from typing import Any


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens // block_size) * block_size


def build_slot_mapping(block_ids: list[int], block_size: int, num_tokens: int) -> list[int]:
    """Map the first num_tokens of a paged allocation to flat KV-cache slots."""
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
        base = block_id * block_size
        slots.extend(range(base, base + block_size))
    return slots[:num_tokens]


def extra_config(kv_transfer_config: Any, key: str, default: Any = None) -> Any:
    getter = getattr(kv_transfer_config, "get_from_extra_config", None)
    if callable(getter):
        return getter(key, default)
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    return extra.get(key, default)


# SamplingParams.extra_args key carrying a request's memory user id, so the
# connector routes each request to its registered memory directly instead of
# prefix-scanning every stored user.
MEMORY_USER_ID_EXTRA_ARG = "locomo_memory_user_id"

# Single home for the KV store backend names and defaults; the CLIs, the
# registry, the connector, and the reporting backfill all reference these.
DEFAULT_KV_STORE_BACKEND = "gpu"
KNOWN_KV_STORE_BACKENDS = ("gpu", "cpu")
DEFAULT_KV_STAGING_SLOTS = 4


def extract_user_id(request: Any, default_user_id: str | None = None) -> str | None:
    user_id = getattr(request, "user", None)
    if user_id:
        return str(user_id)

    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is not None:
        user_id = getattr(sampling_params, "user", None)
        if user_id:
            return str(user_id)
        extra_args = getattr(sampling_params, "extra_args", None)
        if isinstance(extra_args, dict):
            user_id = extra_args.get(MEMORY_USER_ID_EXTRA_ARG)
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
