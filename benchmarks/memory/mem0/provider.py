"""Create Mem0 ``Memory`` objects wired to the benchmark's vector store, embedder, and LLM.

Mem0 resolves vector store providers by name: its factory maps the name to a
class path, and its pydantic ``VectorStoreConfig`` imports
``mem0.configs.vector_stores.<name>`` to validate the provider's settings.
Registering the benchmark's "jasper" provider therefore adds the class path to
the factory, installs a config module under that import path, and adds the
name to the pydantic model's private provider table.
"""

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
from inferscale.v1.embedding.openai import DEFAULT_OPENAI_BASE_URL

JASPER_PROVIDER = "jasper"
JASPER_ADAPTER_CLASS = "benchmarks.memory.mem0.adapter.Mem0JasperVectorStore"
JASPER_CONFIG_MODULE = "mem0.configs.vector_stores.jasper"
VECTOR_BACKENDS = ("jasper", "qdrant")

_registration_lock = threading.Lock()


@dataclass(slots=True, frozen=True)
class Mem0Settings:
    """Everything needed to construct one Mem0 memory for the benchmark."""

    store_root: Path
    vector_config: VectorStoreConfig
    embedding_model: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    memory_llm_model: str
    memory_llm_provider: str = "vllm"
    memory_llm_api_key: str | None = None
    memory_llm_base_url: str | None = None


def create_mem0_memory(settings: Mem0Settings) -> Any:
    """A Mem0 memory whose vector store is the requested local backend."""
    installed_version = importlib.metadata.version("mem0ai")
    if installed_version != MEM0AI_VERSION:
        raise RuntimeError(
            f"This benchmark requires mem0ai=={MEM0AI_VERSION}, found {installed_version}."
        )
    register_mem0_jasper_provider()
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
        "openai_base_url": settings.embedding_base_url or DEFAULT_OPENAI_BASE_URL,
    }
    if settings.embedding_api_key:
        embedder_config["api_key"] = settings.embedding_api_key
    memory_llm_config: dict[str, Any] = {
        "model": memory_llm_model,
        "temperature": MEMORY_LLM_TEMPERATURE,
        "max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
        "vllm_base_url": settings.memory_llm_base_url or DEFAULT_EXTRACTION_LLM_BASE_URL,
    }
    if settings.memory_llm_api_key:
        memory_llm_config["api_key"] = settings.memory_llm_api_key
    vector = settings.vector_config
    return {
        "vector_store": {
            "provider": JASPER_PROVIDER,
            "config": {
                "collection_name": "memories",
                "path": str(settings.store_root),
                "backend": vector.backend,
                "distance": VECTOR_DISTANCE,
                "n_neighbors": vector.n_neighbors,
                "alpha": vector.alpha,
                "workspace_budget": vector.workspace_budget,
                "beam_width": vector.beam_width,
            },
        },
        "embedder": {"provider": "openai", "config": embedder_config},
        "llm": {"provider": settings.memory_llm_provider, "config": memory_llm_config},
        "history_db_path": str(settings.store_root / "history.sqlite"),
    }


def register_mem0_jasper_provider() -> None:
    """Make the provider name resolvable by Mem0's factory and its pydantic config model."""
    from mem0.utils.factory import VectorStoreFactory
    from mem0.vector_stores.configs import VectorStoreConfig as Mem0VectorStoreConfig

    with _registration_lock:
        VectorStoreFactory.provider_to_class[JASPER_PROVIDER] = JASPER_ADAPTER_CLASS
        install_jasper_config_module()
        # pydantic keeps a private attribute's default on the class and copies it
        # into every new instance, so the provider table is extended there.
        provider_table = Mem0VectorStoreConfig.__private_attributes__["_provider_configs"]
        provider_table.default[JASPER_PROVIDER] = "JasperConfig"


def install_jasper_config_module() -> None:
    """Install the pydantic settings model Mem0 imports for the provider."""
    from pydantic import BaseModel, ConfigDict, Field

    class JasperConfig(BaseModel):
        collection_name: str = Field("memories", description="Name of the collection")
        embedding_model_dims: int | None = Field(
            1536, description="Dimensions of the embedding model"
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
    """Close and reject a memory whose adapter did not end up on the requested backend."""
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
