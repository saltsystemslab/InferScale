"""Process-local registry of KV memory stores keyed by connector namespace.

The facade (or a benchmark client) and the in-process vLLM connector are
constructed independently but must share one store, so they meet here:
whichever side asks for a namespace first creates its store, and both pass
the backend explicitly so a mismatch fails fast.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .connector_utils import (
    DEFAULT_KV_STAGING_SLOTS,
    DEFAULT_KV_STORE_BACKEND,
    KNOWN_KV_STORE_BACKENDS,
)
from .memory_store import GPUMemoryStore


@dataclass(slots=True)
class _Namespace:
    store: Any
    backend: str


class _Registry:
    def __init__(self) -> None:
        self._namespaces: dict[str, _Namespace] = {}
        self._lock = threading.Lock()

    def get_or_create(self, namespace: str, backend: str, num_staging_slots: int) -> Any:
        with self._lock:
            entry = self._namespaces.get(namespace)
            if entry is None:
                entry = _Namespace(_create_store(backend, num_staging_slots), backend)
                self._namespaces[namespace] = entry
            return entry

    def get(self, namespace: str) -> Any | None:
        with self._lock:
            entry = self._namespaces.get(namespace)
        return None if entry is None else entry.store

    def drop(self, namespace: str) -> None:
        with self._lock:
            self._namespaces.pop(namespace, None)

    def clear(self) -> None:
        with self._lock:
            self._namespaces.clear()


_REGISTRY = _Registry()


def get_gpu_memory_store(
    namespace: str = "default",
    *,
    backend: str | None = None,
    num_staging_slots: int | None = None,
) -> Any:
    """Return the process-local KV memory store for a connector namespace."""
    resolved = backend or DEFAULT_KV_STORE_BACKEND
    if resolved not in KNOWN_KV_STORE_BACKENDS:
        raise ValueError(
            f"Unknown KV store backend: {resolved!r}; expected one of {KNOWN_KV_STORE_BACKENDS}."
        )
    entry = _REGISTRY.get_or_create(
        namespace, resolved, int(num_staging_slots or DEFAULT_KV_STAGING_SLOTS)
    )
    if backend is not None and entry.backend != resolved:
        raise RuntimeError(
            f"KV store namespace {namespace!r} already exists with backend "
            f"{entry.backend!r}; requested {resolved!r}."
        )
    return entry.store


def register_user_memory(
    namespace: str,
    *,
    user_id: str,
    kv_by_layer: dict[str, Any],
    num_tokens: int,
    token_ids: list[int],
) -> None:
    get_gpu_memory_store(namespace).add_user_memory(
        user_id=user_id, kv_by_layer=kv_by_layer, num_tokens=num_tokens, token_ids=token_ids
    )


def remove_user_memory(namespace: str, user_id: str) -> bool:
    store = _REGISTRY.get(namespace)
    if store is None:
        return False
    return bool(store.remove_user_memory(user_id))


def clear_namespace(namespace: str) -> None:
    store = _REGISTRY.get(namespace)
    if store is None:
        return
    for user_id in list(store.get_all_user_ids()):
        store.remove_user_memory(user_id)


def drop_namespace(namespace: str) -> None:
    clear_namespace(namespace)
    _REGISTRY.drop(namespace)


def namespace_stats(namespace: str) -> dict[str, Any]:
    store = _REGISTRY.get(namespace)
    if store is None:
        return {"num_users": 0, "total_tokens": 0, "total_gpu_mb": 0.0, "total_host_mb": 0.0}
    return dict(store.get_stats())


def _create_store(backend: str, num_staging_slots: int) -> Any:
    if backend == "cpu":
        from .pinned_memory_store import CpuPinnedMemoryStore

        return CpuPinnedMemoryStore(num_staging_slots=num_staging_slots)
    return GPUMemoryStore()
