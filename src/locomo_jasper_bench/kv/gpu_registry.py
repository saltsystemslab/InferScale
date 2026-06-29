from __future__ import annotations

from typing import Any

from .gpu_memory_store_loader import load_gpu_memory_store_class

_STORES: dict[str, Any] = {}


def get_gpu_memory_store(namespace: str = "default") -> Any:
    """Return the process-local GPU memory store for a connector namespace."""
    if namespace not in _STORES:
        try:
            gpu_memory_store_cls = load_gpu_memory_store_class()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Could not load vendored ai-memory-code GPUMemoryStore. Ensure src/ai-memory-code "
                "contains the required files and the remote environment has torch/vLLM dependencies installed."
            ) from exc
        _STORES[namespace] = gpu_memory_store_cls()
    return _STORES[namespace]


def register_user_memory(
    namespace: str,
    *,
    user_id: str,
    kv_by_layer: dict[str, Any],
    num_tokens: int,
    token_ids: list[int],
    memory_text: str = "",
) -> None:
    get_gpu_memory_store(namespace).add_user_memory(
        user_id=user_id,
        kv_by_layer=kv_by_layer,
        num_tokens=num_tokens,
        token_ids=token_ids,
        memory_text=memory_text,
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


def namespace_stats(namespace: str) -> dict[str, Any]:
    store = _STORES.get(namespace)
    if store is None:
        return {"num_users": 0, "total_tokens": 0, "total_gpu_mb": 0.0}
    return dict(store.get_stats())
