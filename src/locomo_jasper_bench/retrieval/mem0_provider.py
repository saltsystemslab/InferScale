from __future__ import annotations

import importlib.metadata
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from ..protocol import MEM0AI_VERSION, MEMORY_EXTRACTION_MAX_TOKENS
from ..runtime_paths import default_mem0_dir_string
from ..vector_types import VECTOR_DISTANCE, VectorStoreConfig

# Extraction sampling temperature. Part of the fact-catalog and LLM-cache
# identity: changing it must not silently replay catalogs extracted at the
# old value.
MEMORY_LLM_TEMPERATURE = 0.0
_MEM0_PROVIDER_REGISTRATION_LOCK = threading.Lock()


def create_mem0_memory(
    *,
    store_root: str | Path,
    vector_config: VectorStoreConfig,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
    memory_llm_provider: str = "vllm",
    memory_llm_model: str,
    memory_llm_api_key: str | None = None,
    memory_llm_base_url: str | None = None,
) -> Any:
    installed_version = importlib.metadata.version("mem0ai")
    if installed_version != MEM0AI_VERSION:
        raise RuntimeError(
            f"This benchmark requires mem0ai=={MEM0AI_VERSION}, found {installed_version}."
        )
    os.environ.setdefault("MEM0_DIR", default_mem0_dir_string())
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    register_mem0_jasper_provider()
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError("Install the mem0ai package to run Mem0 retrieval.") from exc

    store_root = Path(store_root)
    store_root.mkdir(parents=True, exist_ok=True)
    mem0_config = build_mem0_config(
        store_root=store_root,
        vector_config=vector_config,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        memory_llm_provider=memory_llm_provider,
        memory_llm_model=memory_llm_model,
        memory_llm_api_key=memory_llm_api_key,
        memory_llm_base_url=memory_llm_base_url,
    )
    if hasattr(Memory, "from_config"):
        memory = Memory.from_config(mem0_config)
    else:
        from mem0.configs.base import MemoryConfig

        memory = Memory(MemoryConfig(**mem0_config))
    _validate_resolved_backend(memory, requested_backend=vector_config.backend)
    return memory


def build_mem0_config(
    *,
    store_root: str | Path,
    vector_config: VectorStoreConfig,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
    memory_llm_provider: str = "vllm",
    memory_llm_model: str,
    memory_llm_api_key: str | None = None,
    memory_llm_base_url: str | None = None,
) -> dict[str, Any]:
    if memory_llm_provider != "vllm":
        raise ValueError(f"Unsupported Mem0 LLM provider: {memory_llm_provider!r}.")
    memory_llm_model = memory_llm_model.strip()
    if not memory_llm_model:
        raise ValueError("Mem0 LLM model must not be empty.")

    embedder_config: dict[str, Any] = {"model": embedding_model}
    if embedding_api_key:
        embedder_config["api_key"] = embedding_api_key
    if embedding_base_url:
        embedder_config["openai_base_url"] = embedding_base_url

    memory_llm_config: dict[str, Any] = {
        "model": memory_llm_model,
        "temperature": MEMORY_LLM_TEMPERATURE,
        "max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
    }
    if memory_llm_api_key:
        memory_llm_config["api_key"] = memory_llm_api_key
    if memory_llm_base_url:
        memory_llm_config["vllm_base_url"] = memory_llm_base_url

    store_root = Path(store_root)
    return {
        "vector_store": {
            "provider": "jasper",
            "config": {
                "collection_name": "memories",
                "path": str(store_root),
                "backend": vector_config.backend,
                "distance": VECTOR_DISTANCE,
                "n_neighbors": vector_config.n_neighbors,
                "alpha": vector_config.alpha,
                "workspace_budget": vector_config.workspace_budget,
                "beam_width": vector_config.beam_width,
            },
        },
        "embedder": {"provider": "openai", "config": embedder_config},
        "llm": {"provider": memory_llm_provider, "config": memory_llm_config},
        "history_db_path": str(store_root / "history.sqlite"),
    }


def register_mem0_jasper_provider() -> None:
    try:
        from mem0.utils.factory import VectorStoreFactory
        from mem0.vector_stores.configs import VectorStoreConfig as Mem0VectorStoreConfig
    except ImportError as exc:
        raise RuntimeError("Install the mem0ai package to register the Jasper Mem0 provider.") from exc

    with _MEM0_PROVIDER_REGISTRATION_LOCK:
        VectorStoreFactory.provider_to_class["jasper"] = (
            "locomo_jasper_bench.retrieval.mem0_adapter.Mem0JasperVectorStore"
        )
        _install_jasper_config_module()
        _patch_mem0_vector_config_registry(Mem0VectorStoreConfig)


def _install_jasper_config_module() -> None:
    try:
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError("mem0ai requires pydantic; install mem0ai to configure the Jasper provider.") from exc

    class JasperConfig(BaseModel):
        collection_name: str = Field("memories", description="Name of the collection")
        embedding_model_dims: int | None = Field(1536, description="Dimensions of the embedding model")
        path: str = Field(
            default_factory=default_mem0_dir_string,
            description="Path for the Jasper vector store",
        )
        backend: Literal["jasper", "qdrant"] = Field("jasper", description="Concrete vector store backend")
        distance: Literal["ip"] = Field(VECTOR_DISTANCE, description="Distance metric")
        n_neighbors: int = Field(64, description="Jasper graph neighbor count")
        alpha: float = Field(1.0, description="Jasper graph alpha")
        workspace_budget: str = Field("10GB", description="Jasper graph build workspace budget")
        beam_width: int = Field(64, description="Jasper search beam width")

        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    module_name = "mem0.configs.vector_stores.jasper"
    module = sys.modules.get(module_name) or types.ModuleType(module_name)
    module.JasperConfig = JasperConfig
    sys.modules[module_name] = module


def _patch_mem0_vector_config_registry(mem0_vector_config_cls: Any) -> None:
    registry = getattr(mem0_vector_config_cls, "_provider_configs", None)
    if isinstance(registry, dict):
        registry["jasper"] = "JasperConfig"

    private_attrs = getattr(mem0_vector_config_cls, "__private_attributes__", {})
    private_attr = private_attrs.get("_provider_configs") if isinstance(private_attrs, dict) else None
    default = getattr(private_attr, "default", None)
    if isinstance(default, dict):
        default["jasper"] = "JasperConfig"


def resolved_mem0_backend(memory: Any) -> str:
    vector_store = getattr(memory, "vector_store", None)
    config = getattr(vector_store, "config", None)
    backend = getattr(config, "backend", None)
    if backend not in {"jasper", "qdrant"}:
        raise RuntimeError(f"Mem0 vector store did not expose a valid resolved backend: {backend!r}.")
    return str(backend)


def _validate_resolved_backend(memory: Any, *, requested_backend: str) -> None:
    from .jasper_vector_store import JasperVectorStore
    from .qdrant_vector_store import QdrantVectorStore

    if requested_backend not in {"jasper", "qdrant"}:
        raise ValueError(f"Unsupported requested vector backend: {requested_backend!r}.")
    vector_store = getattr(memory, "vector_store", None)
    concrete_store = getattr(vector_store, "store", None)
    resolved_backend = resolved_mem0_backend(memory)
    expected_store_type = JasperVectorStore if requested_backend == "jasper" else QdrantVectorStore
    if resolved_backend != requested_backend or not isinstance(concrete_store, expected_store_type):
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()
        raise RuntimeError(
            "Mem0 vector backend mismatch: "
            f"requested={requested_backend!r} resolved={resolved_backend!r} "
            f"store={type(concrete_store).__name__}."
        )
    logger.info(
        "Mem0 vector backend resolved requested={} resolved={} store={}",
        requested_backend,
        resolved_backend,
        type(concrete_store).__name__,
    )
