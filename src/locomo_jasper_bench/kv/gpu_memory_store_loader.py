from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from typing import Any

from .submodule import repo_root, require_ai_memory_submodule

_MODULE_NAME = "_locomo_jasper_bench_ai_memory_gpu_memory_store"
_MODULE: ModuleType | None = None


def load_gpu_memory_store_class() -> type[Any]:
    """Load ai-memory-code's GPUMemoryStore without importing its package."""
    global _MODULE
    if _MODULE is None:
        require_ai_memory_submodule()
        module_path = repo_root() / "ai-memory-code" / "memory_connector" / "gpu_memory_store.py"
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load GPUMemoryStore module from {module_path}.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        _MODULE = module

    cls = getattr(_MODULE, "GPUMemoryStore", None)
    if cls is None:
        raise RuntimeError("ai-memory-code gpu_memory_store.py does not define GPUMemoryStore.")
    return cls
