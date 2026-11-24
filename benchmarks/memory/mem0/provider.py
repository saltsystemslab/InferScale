from __future__ import annotations

import importlib.metadata
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from benchmarks.common.config import DEFAULT_EXTRACTION_LLM_BASE_URL
from benchmarks.common.paths import mem0_dir_from_environment
from benchmarks.common.vector_types import VECTOR_DISTANCE, VectorStoreConfig
from benchmarks.memory.protocol import (
    MEM0AI_VERSION,
    MEMORY_EXTRACTION_MAX_TOKENS,
    MEMORY_LLM_TEMPERATURE,
)

JASPER_PROVIDER = "jasper"
JASPER_ADAPTER_CLASS = "benchmarks.memory.mem0.adapter.Mem0JasperVectorStore"
JASPER_CONFIG_MODULE = "mem0.configs.vector_stores.jasper"
LOCAL_EMBEDDER_PROVIDER = "huggingface"
LOCAL_EMBEDDER_CLASS = "benchmarks.memory.mem0.local_embedder.Mem0LocalEmbedder"
VECTOR_BACKENDS = ("jasper", "qdrant")

_registration_lock = threading.Lock()


@dataclass(slots=True, frozen=True)
class Mem0Settings:
    store_root: Path
    vector_config: VectorStoreConfig
    embedding_model: str
    embedding_device: str
    memory_llm_model: str
    memory_llm_provider: str = "vllm"
    memory_llm_api_key: str | None = None
    memory_llm_base_url: str | None = None


def create_mem0_memory(settings: Mem0Settings) -> Any:
    installed_version = importlib.metadata.version("mem0ai")
    if installed_version != MEM0AI_VERSION:
        raise RuntimeError(
            f"This benchmark requires mem0ai=={MEM0AI_VERSION}, found {installed_version}."
        )
    register_mem0_jasper_provider()
    register_mem0_local_embedder()
    from mem0 import Memory

    settings.store_root.mkdir(parents=True, exist_ok=True)
    memory = Memory.from_config(build_mem0_config(settings))
    validate_resolved_backend(memory, requested_backend=settings.vector_config.backend)
    return memory


def build_mem0_config(settings: Mem0Settings) -> dict[str, Any]:
    if settings.memory_llm_provider != "vllm":
        raise ValueError(f"Unsupported Mem0 LLM provider: {settings.memory_llm_provider!r}.")
    memory_llm_model = settings.memory_llm_model.strip()
    if not memory_llm_model:
        raise ValueError("Mem0 LLM model must not be empty.")

    embedder_config: dict[str, Any] = {
        "model": settings.embedding_model,
        "model_kwargs": {"device": settings.embedding_device},
    }
    memory_llm_config: dict[str, Any] = {
        "model": memory_llm_model,
        "temperature": MEMORY_LLM_TEMPERATURE,
        "max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
        "vllm_base_url": settings.memory_llm_base_url or DEFAULT_EXTRACTION_LLM_BASE_URL,
    }
    if settings.memory_llm_api_key:
        memory_llm_config["api_key"] = settings.memory_llm_api_key
    vector_config = settings.vector_config
    return {
        "vector_store": {
            "provider": JASPER_PROVIDER,
            "config": {
                "collection_name": "memories",
                "path": str(settings.store_root),
                "backend": vector_config.backend,
                "distance": VECTOR_DISTANCE,
                "n_neighbors": vector_config.n_neighbors,
                "alpha": vector_config.alpha,
                "workspace_budget": vector_config.workspace_budget,
                "beam_width": vector_config.beam_width,
            },
        },
        "embedder": {"provider": LOCAL_EMBEDDER_PROVIDER, "config": embedder_config},
        "llm": {"provider": settings.memory_llm_provider, "config": memory_llm_config},
        "history_db_path": str(settings.store_root / "history.sqlite"),
    }


def register_mem0_jasper_provider() -> None:
    from mem0.utils.factory import VectorStoreFactory
    from mem0.vector_stores.configs import VectorStoreConfig as Mem0VectorStoreConfig

    with _registration_lock:
        VectorStoreFactory.provider_to_class[JASPER_PROVIDER] = JASPER_ADAPTER_CLASS
        install_jasper_config_module()
        provider_table = Mem0VectorStoreConfig.__private_attributes__["_provider_configs"]
        provider_table.default[JASPER_PROVIDER] = "JasperConfig"


def register_mem0_local_embedder() -> None:
    from mem0.utils.factory import EmbedderFactory

    with _registration_lock:
        EmbedderFactory.provider_to_class[LOCAL_EMBEDDER_PROVIDER] = LOCAL_EMBEDDER_CLASS


def install_jasper_config_module() -> None:
    from pydantic import BaseModel, ConfigDict, Field

    class JasperConfig(BaseModel):
        collection_name: str = Field("memories", description="Name of the collection")
        embedding_model_dims: int | None = Field(
            None, description="Dimensions of the embedding model"
        )
        path: str = Field(
            default_factory=lambda: str(mem0_dir_from_environment()),
            description="Path for the vector store",
        )
        backend: Literal["jasper", "qdrant"] = Field(
            "jasper", description="Concrete vector store backend"
        )
        distance: Literal["ip"] = Field(VECTOR_DISTANCE, description="Distance metric")
        n_neighbors: int = Field(64, description="Jasper graph neighbor count")
        alpha: float = Field(1.0, description="Jasper graph alpha")
        workspace_budget: str = Field("10GB", description="Jasper graph build workspace budget")
        beam_width: int = Field(64, description="Jasper search beam width")

        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    module = sys.modules.get(JASPER_CONFIG_MODULE) or types.ModuleType(JASPER_CONFIG_MODULE)
    module.JasperConfig = JasperConfig
    sys.modules[JASPER_CONFIG_MODULE] = module


def resolved_mem0_backend(memory: Any) -> str:
    backend = memory.vector_store.config.backend
    if backend not in VECTOR_BACKENDS:
        raise RuntimeError(
            f"Mem0 vector store did not expose a valid resolved backend: {backend!r}."
        )
    return str(backend)


def validate_resolved_backend(memory: Any, *, requested_backend: str) -> None:
    from benchmarks.memory.qdrant_index import QdrantVectorStore
    from inferscale.v1.index.jasper import JasperIndex

    if requested_backend not in VECTOR_BACKENDS:
        raise ValueError(f"Unsupported requested vector backend: {requested_backend!r}.")
    vector_store = memory.vector_store
    concrete_store = vector_store.store
    resolved_backend = resolved_mem0_backend(memory)
    expected_store_type = JasperIndex if requested_backend == "jasper" else QdrantVectorStore
    if resolved_backend != requested_backend or not isinstance(concrete_store, expected_store_type):
        vector_store.close()
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
