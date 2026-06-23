from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gpu_chunk_store import GpuChunkPlan, GpuSampleChunkStore


@dataclass(slots=True)
class UserMemory:
    kv_by_layer: dict[str, Any]
    num_tokens: int
    token_ids: list[int] | None = None
    memory_text: str = ""


class StrictGpuKVStore:
    """Process-local strict GPU store for composite memories and chunk plans."""

    def __init__(self) -> None:
        self._memories: dict[str, UserMemory] = {}
        self._sample_stores: dict[str, GpuSampleChunkStore] = {}
        self._plans: dict[str, GpuChunkPlan] = {}

    def add_user_memory(
        self,
        user_id: str,
        kv_by_layer: dict[str, Any],
        num_tokens: int,
        token_ids: list[int] | None = None,
        memory_text: str = "",
    ) -> None:
        self._memories[user_id] = UserMemory(
            kv_by_layer={
                layer_name: tensor.contiguous()
                for layer_name, tensor in kv_by_layer.items()
            },
            num_tokens=num_tokens,
            token_ids=token_ids,
            memory_text=memory_text,
        )

    def get_user_memory(self, user_id: str) -> UserMemory | None:
        return self._memories.get(user_id)

    def remove_user_memory(self, user_id: str) -> bool:
        removed = False
        if user_id in self._memories:
            memory = self._memories.pop(user_id)
            del memory.kv_by_layer
            removed = True
        if user_id in self._plans:
            self._plans.pop(user_id, None)
            removed = True
        return removed

    def get_all_user_ids(self) -> list[str]:
        return sorted({*self._memories.keys(), *self._plans.keys()})

    def add_sample_store(self, sample_id: str, store: GpuSampleChunkStore) -> None:
        self.remove_sample_store(sample_id)
        self._sample_stores[sample_id] = store

    def get_sample_store(self, sample_id: str) -> GpuSampleChunkStore | None:
        return self._sample_stores.get(sample_id)

    def remove_sample_store(self, sample_id: str) -> bool:
        store = self._sample_stores.pop(sample_id, None)
        if store is None:
            return False
        store.close()
        return True

    def add_chunk_plan(self, plan: GpuChunkPlan) -> None:
        self._plans[plan.plan_id] = plan

    def get_chunk_plan(self, plan_id: str) -> GpuChunkPlan | None:
        return self._plans.get(plan_id)

    def remove_chunk_plan(self, plan_id: str) -> bool:
        return self._plans.pop(plan_id, None) is not None

    def clear(self) -> None:
        for user_id in list(self.get_all_user_ids()):
            self.remove_user_memory(user_id)
        for sample_id in list(self._sample_stores):
            self.remove_sample_store(sample_id)

    def get_stats(self) -> dict[str, Any]:
        composite_bytes = 0
        composite_tokens = 0
        for memory in self._memories.values():
            composite_tokens += memory.num_tokens
            for tensor in memory.kv_by_layer.values():
                composite_bytes += _tensor_nbytes(tensor)

        sample_bytes = 0.0
        sample_tokens = 0
        sample_chunks = 0
        for store in self._sample_stores.values():
            stats = store.get_stats()
            sample_bytes += float(stats.get("sample_gpu_mb", 0.0)) * 1024 * 1024
            sample_tokens += int(stats.get("sample_chunk_tokens", 0))
            sample_chunks += int(stats.get("sample_chunks", 0))

        total_bytes = composite_bytes + int(sample_bytes)
        return {
            "num_users": len(self._memories) + len(self._plans),
            "num_plans": len(self._plans),
            "num_sample_stores": len(self._sample_stores),
            "sample_chunks": sample_chunks,
            "total_tokens": composite_tokens + sample_tokens,
            "composite_tokens": composite_tokens,
            "sample_chunk_tokens": sample_tokens,
            "total_gpu_mb": total_bytes / (1024 * 1024),
            "composite_gpu_mb": composite_bytes / (1024 * 1024),
            "sample_gpu_mb": sample_bytes / (1024 * 1024),
        }


_STORES: dict[str, StrictGpuKVStore] = {}


def get_gpu_memory_store(namespace: str = "default") -> StrictGpuKVStore:
    """Return the process-local strict GPU store for a connector namespace."""
    if namespace not in _STORES:
        _STORES[namespace] = StrictGpuKVStore()
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


def register_sample_store(namespace: str, *, sample_id: str, store: GpuSampleChunkStore) -> None:
    get_gpu_memory_store(namespace).add_sample_store(sample_id, store)


def remove_sample_store(namespace: str, sample_id: str) -> bool:
    store = _STORES.get(namespace)
    if store is None:
        return False
    return store.remove_sample_store(sample_id)


def register_chunk_plan(namespace: str, plan: GpuChunkPlan) -> None:
    get_gpu_memory_store(namespace).add_chunk_plan(plan)


def remove_user_memory(namespace: str, user_id: str) -> bool:
    store = _STORES.get(namespace)
    if store is None:
        return False
    return bool(store.remove_user_memory(user_id))


def namespace_stats(namespace: str) -> dict[str, Any]:
    store = _STORES.get(namespace)
    if store is None:
        return {
            "num_users": 0,
            "num_plans": 0,
            "num_sample_stores": 0,
            "sample_chunks": 0,
            "total_tokens": 0,
            "composite_tokens": 0,
            "sample_chunk_tokens": 0,
            "total_gpu_mb": 0.0,
            "composite_gpu_mb": 0.0,
            "sample_gpu_mb": 0.0,
        }
    return store.get_stats()


def clear_namespace(namespace: str) -> None:
    store = _STORES.get(namespace)
    if store is None:
        return
    store.clear()


def drop_namespace(namespace: str) -> None:
    clear_namespace(namespace)
    _STORES.pop(namespace, None)


def _tensor_nbytes(tensor: Any) -> int:
    value = getattr(tensor, "nbytes", None)
    if value is not None:
        return int(value)
    element_size = getattr(tensor, "element_size", None)
    nelement = getattr(tensor, "nelement", None)
    if callable(element_size) and callable(nelement):
        return int(element_size() * nelement())
    return 0
