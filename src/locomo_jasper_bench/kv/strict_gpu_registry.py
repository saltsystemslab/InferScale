from __future__ import annotations

from copy import deepcopy
from typing import Any

from .gpu_memory_store_loader import load_gpu_memory_store_class

_STORES: dict[str, Any] = {}
_DIAGNOSTICS: dict[str, dict[str, Any]] = {}


def _default_diagnostics() -> dict[str, Any]:
    return {
        "connector_init_count": 0,
        "connector_match_attempts": 0,
        "connector_match_hits": 0,
        "connector_match_misses": 0,
        "connector_update_state_calls": 0,
        "connector_build_meta_calls": 0,
        "connector_metadata_loads": 0,
        "connector_start_load_calls": 0,
        "connector_injected_tokens": 0,
        "connector_missing_memory_loads": 0,
        "connector_missing_layer_loads": 0,
        "connector_block_size": 0,
        "connector_last_role": "",
        "connector_last_user_id": "",
        "connector_last_miss_reason": "",
        "connector_last_mismatch_index": -1,
        "connector_last_prompt_tokens": 0,
        "connector_last_raw_memory_tokens": 0,
        "connector_last_aligned_tokens": 0,
        "connector_last_new_tokens": 0,
        "connector_last_num_computed_tokens": 0,
        "connector_last_request_id": "",
    }


def _diagnostics_for(namespace: str) -> dict[str, Any]:
    if namespace not in _DIAGNOSTICS:
        _DIAGNOSTICS[namespace] = _default_diagnostics()
    return _DIAGNOSTICS[namespace]


def update_namespace_diagnostics(
    namespace: str,
    *,
    increments: dict[str, int] | None = None,
    values: dict[str, Any] | None = None,
) -> None:
    diagnostics = _diagnostics_for(namespace)
    for key, amount in (increments or {}).items():
        diagnostics[key] = int(diagnostics.get(key, 0) or 0) + int(amount)
    for key, value in (values or {}).items():
        diagnostics[key] = value


def reset_namespace_diagnostics(namespace: str) -> None:
    existing = _DIAGNOSTICS.get(namespace, {})
    reset = _default_diagnostics()
    for key in ("connector_init_count", "connector_block_size", "connector_last_role"):
        if key in existing:
            reset[key] = existing[key]
    _DIAGNOSTICS[namespace] = reset


def namespace_diagnostics(namespace: str) -> dict[str, Any]:
    return deepcopy(_diagnostics_for(namespace))


def get_gpu_memory_store(namespace: str = "default") -> Any:
    """Return the process-local GPU memory store for a connector namespace."""
    if namespace not in _STORES:
        try:
            gpu_memory_store_cls = load_gpu_memory_store_class()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Could not load ai-memory-code GPUMemoryStore. Ensure the submodule exists "
                "and the remote environment has torch/vLLM dependencies installed."
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
    store = _STORES.pop(namespace, None)
    _DIAGNOSTICS.pop(namespace, None)
    if store is None:
        return
    for user_id in list(store.get_all_user_ids()):
        store.remove_user_memory(user_id)


def namespace_stats(namespace: str) -> dict[str, Any]:
    store = _STORES.get(namespace)
    if store is None:
        return {"num_users": 0, "total_tokens": 0, "total_gpu_mb": 0.0}
    return dict(store.get_stats())
