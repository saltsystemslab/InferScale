from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

from ..vector_types import VectorStoreConfig


def create_mem0_memory(
    *,
    store_root: str | Path,
    vector_config: VectorStoreConfig,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
) -> Any:
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
    )
    if hasattr(Memory, "from_config"):
        return Memory.from_config(mem0_config)

    from mem0.configs.base import MemoryConfig

    return Memory(MemoryConfig(**mem0_config))


def build_mem0_config(
    *,
    store_root: str | Path,
    vector_config: VectorStoreConfig,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
) -> dict[str, Any]:
    embedder_config: dict[str, Any] = {"model": embedding_model}
    if embedding_api_key:
        embedder_config["api_key"] = embedding_api_key
    if embedding_base_url:
        embedder_config["openai_base_url"] = embedding_base_url

    store_root = Path(store_root)
    return {
        "vector_store": {
            "provider": "jasper",
            "config": {
                "collection_name": "memories",
                "path": str(store_root),
                "backend": vector_config.backend,
                "distance": vector_config.distance,
                "n_neighbors": vector_config.n_neighbors,
                "alpha": vector_config.alpha,
                "workspace_budget": vector_config.workspace_budget,
                "beam_width": vector_config.beam_width,
            },
        },
        "embedder": {"provider": "openai", "config": embedder_config},
        "history_db_path": str(store_root / "history.sqlite"),
    }


def register_mem0_jasper_provider() -> None:
    try:
        from mem0.utils.factory import VectorStoreFactory
        from mem0.vector_stores.configs import VectorStoreConfig as Mem0VectorStoreConfig
    except ImportError as exc:
        raise RuntimeError("Install the mem0ai package to register the Jasper Mem0 provider.") from exc

    VectorStoreFactory.provider_to_class["jasper"] = "locomo_jasper_bench.mem0_adapter.Mem0JasperVectorStore"
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
            default_factory=_default_mem0_dir_string,
            description="Path for the Jasper vector store",
        )
        distance: str = Field("ip", description="Distance metric")
        n_neighbors: int = Field(64, description="Jasper graph neighbor count")
        alpha: float = Field(1.0, description="Jasper graph alpha")
        workspace_budget: str = Field("10GB", description="Jasper graph build workspace budget")
        beam_width: int = Field(64, description="Jasper search beam width")

        model_config = ConfigDict(arbitrary_types_allowed=True)

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


def _default_mem0_dir_string() -> str:
    if "MEM0_DIR" in os.environ:
        return os.environ["MEM0_DIR"]
    if "BENCHMARK_CACHE_ROOT" in os.environ:
        cache_root = Path(os.environ["BENCHMARK_CACHE_ROOT"])
    elif "SCRATCH_ROOT" in os.environ:
        cache_root = Path(os.environ["SCRATCH_ROOT"]) / "cache"
    else:
        cache_root = Path(".cache")
    return str(cache_root / "mem0")
