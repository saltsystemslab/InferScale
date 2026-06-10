from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from .strict_gpu_registry import get_gpu_memory_store
from .submodule import require_ai_memory_submodule

require_ai_memory_submodule()

from memory_connector.gpu_memory_store import GPUMemoryStore  # noqa: E402
from memory_connector.memory_kv_connector import MemoryKVConnector as _MemoryKVConnector  # noqa: E402


@contextmanager
def _forbid_disk_loads() -> Any:
    original = GPUMemoryStore.load_all_from_disk

    def rejected(*_: Any, **__: Any) -> int:
        raise RuntimeError(
            "Strict GPU KV mode does not allow loading memory KV from disk or CPU. "
            "Register composed KV tensors through locomo_jasper_bench.kv.strict_gpu_registry instead."
        )

    GPUMemoryStore.load_all_from_disk = rejected
    try:
        yield
    finally:
        GPUMemoryStore.load_all_from_disk = original


class MemoryKVConnector(_MemoryKVConnector):
    """MemoryKVConnector variant backed by a process-local GPU store.

    The base connector already handles vLLM scheduler/worker metadata and
    paged-cache scatter. This wrapper only swaps the store for a namespace
    shared with the benchmark process and rejects disk-backed `memory_path`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _forbid_disk_loads():
            super().__init__(*args, **kwargs)

        namespace = self._kv_transfer_config.get_from_extra_config(  # type: ignore[attr-defined]
            "memory_namespace",
            "default",
        )
        self._strict_memory_namespace = str(namespace)
        self._memory_store = get_gpu_memory_store(self._strict_memory_namespace)


__all__ = ["MemoryKVConnector"]

