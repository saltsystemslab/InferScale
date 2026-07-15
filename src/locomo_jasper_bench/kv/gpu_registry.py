from __future__ import annotations

import os
from typing import Any

from .connector_utils import (
    DEFAULT_KV_STAGING_SLOTS,
    DEFAULT_KV_STORE_BACKEND,
    KNOWN_KV_STORE_BACKENDS,
)
from .gpu_memory_store import GPUMemoryStore

_STORES: dict[str, Any] = {}
_BACKENDS: dict[str, str] = {}


def get_gpu_memory_store(
    namespace: str = "default",
    *,
    backend: str | None = None,
    num_staging_slots: int | None = None,
) -> Any:
    """Return the process-local KV memory store for a connector namespace.

    The driver (worker/answer client) and the in-process connector both
    resolve the same namespace; whichever runs first creates the store, so
    both pass the backend explicitly and a mismatch fails fast.
    """
    resolved = backend or os.environ.get("LOCOMO_KV_STORE_BACKEND") or DEFAULT_KV_STORE_BACKEND
    if resolved not in KNOWN_KV_STORE_BACKENDS:
        raise ValueError(f"Unknown KV store backend: {resolved!r}; expected one of {KNOWN_KV_STORE_BACKENDS}.")

    if namespace in _STORES:
        existing = _BACKENDS.get(namespace, DEFAULT_KV_STORE_BACKEND)
        if backend is not None and existing != resolved:
            raise RuntimeError(
                f"KV store namespace {namespace!r} already exists with backend "
                f"{existing!r}; requested {resolved!r}."
            )
        return _STORES[namespace]

    if resolved == "cpu":
        from .cpu_memory_store import CpuPinnedMemoryStore

        store: Any = CpuPinnedMemoryStore(num_staging_slots=int(num_staging_slots or DEFAULT_KV_STAGING_SLOTS))
    else:
        store = GPUMemoryStore()
    _STORES[namespace] = store
    _BACKENDS[namespace] = resolved
    return store


def register_user_memory(
    namespace: str,
    *,
    user_id: str,
    kv_by_layer: dict[str, Any],
    num_tokens: int,
    token_ids: list[int],
) -> None:
    get_gpu_memory_store(namespace).add_user_memory(
        user_id=user_id,
        kv_by_layer=kv_by_layer,
        num_tokens=num_tokens,
        token_ids=token_ids,
    )


def remove_user_memory(namespace: str, user_id: str) -> bool:
    store = _STORES.get(namespace)
    if store is None:
        return False
    return bool(store.remove_user_memory(user_id))


def clear_namespace(namespace: str) -> None:
    store = _STORES.get(namespace)
    if store is None:
        return
    for user_id in list(store.get_all_user_ids()):
        store.remove_user_memory(user_id)


def drop_namespace(namespace: str) -> None:
    clear_namespace(namespace)
    _STORES.pop(namespace, None)
    _BACKENDS.pop(namespace, None)


def namespace_stats(namespace: str) -> dict[str, Any]:
    store = _STORES.get(namespace)
    if store is None:
        return {"num_users": 0, "total_tokens": 0, "total_gpu_mb": 0.0, "total_host_mb": 0.0}
    return dict(store.get_stats())


def namespace_bench_summary(namespace: str) -> dict[str, Any]:
    """Transfer metrics for stores that track them; empty for the GPU store."""
    store = _STORES.get(namespace)
    if store is None:
        return {}
    return dict(store.get_bench_summary())


def reset_namespace_bench_metrics(namespace: str) -> None:
    store = _STORES.get(namespace)
    if store is not None:
        store.reset_bench_metrics()


def namespace_last_transfer(namespace: str) -> Any:
    """Most recent TransferRecord, or None for the GPU store."""
    store = _STORES.get(namespace)
    if store is None:
        return None
    return store.last_transfer_record()


def namespace_transfer_count(namespace: str) -> int:
    """Lifetime transfer count; 0 for the GPU store."""
    store = _STORES.get(namespace)
    if store is None:
        return 0
    return int(store.transfer_count())
