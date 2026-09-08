"""Configuration of the InferScale v1 API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .index.jasper import MAX_JASPER_BEAM_WIDTH, JasperIndexConfig
from .kv.connector_utils import (
    DEFAULT_KV_STAGING_SLOTS,
    DEFAULT_KV_STORE_BACKEND,
    KNOWN_KV_STORE_BACKENDS,
)

DEFAULT_CONNECTOR_MODULE = "inferscale.v1.kv.connector"
KV_DTYPES = ("bfloat16", "bf16", "float16", "fp16", "half", "float32", "fp32")


@dataclass(slots=True)
class EngineConfig:
    """In-process vLLM engine settings."""

    gpu_memory_utilization: float = 0.40
    max_model_len: int = 32768
    block_size: int = 16
    enable_prefix_caching: bool = True
    connector_module: str = DEFAULT_CONNECTOR_MODULE


@dataclass(slots=True)
class KVConfig:
    """Chunk encoder and chunk store settings."""

    dtype: str = "bfloat16"
    device: str = "cuda:0"
    max_position: int = 32768
    store_backend: str = DEFAULT_KV_STORE_BACKEND
    staging_slots: int = DEFAULT_KV_STAGING_SLOTS


@dataclass(slots=True)
class EmbeddingConfig:
    """Embedding model; the API key comes from OPENAI_API_KEY."""

    model: str = "text-embedding-3-small"
    base_url: str | None = None
    batch_size: int = 128


@dataclass(slots=True)
class PromptConfig:
    """Text around the injected context and between chunks."""

    system_prompt: str = "You are a helpful assistant. Answer using the retrieved context below.\n\n"
    empty_context_text: str = "(No relevant context found)\n"
    chunk_separator: str = "\n"


@dataclass(slots=True)
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(slots=True)
class InferScaleConfig:
    model: str
    top_k: int = 10
    engine: EngineConfig = field(default_factory=EngineConfig)
    kv: KVConfig = field(default_factory=KVConfig)
    index: JasperIndexConfig = field(default_factory=JasperIndexConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        validate_config(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferScaleConfig":
        return _build(cls, data, "inferscale")

    @classmethod
    def from_json(cls, path: str | Path) -> "InferScaleConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"InferScale config must be a JSON object: {path}")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NESTED_SECTIONS: dict[str, type] = {
    "engine": EngineConfig,
    "kv": KVConfig,
    "index": JasperIndexConfig,
    "embedding": EmbeddingConfig,
    "prompt": PromptConfig,
    "generation": GenerationConfig,
}


def _build(cls: type, data: Mapping[str, Any], where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ValueError(f"{where} must be a JSON object.")
    names = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{where} has unknown keys: {', '.join(unknown)}.")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        nested = _NESTED_SECTIONS.get(name) if cls is InferScaleConfig else None
        kwargs[name] = _build(nested, value, f"{where}.{name}") if nested is not None else value
    return cls(**kwargs)


def validate_config(config: InferScaleConfig) -> None:
    if not str(config.model).strip():
        raise ValueError("model must be a non-empty model id or path.")
    if config.top_k < 1:
        raise ValueError("top_k must be >= 1.")
    engine = config.engine
    if engine.block_size < 1:
        raise ValueError("engine.block_size must be >= 1.")
    if engine.max_model_len < 1:
        raise ValueError("engine.max_model_len must be >= 1.")
    if not 0 < engine.gpu_memory_utilization < 1:
        raise ValueError("engine.gpu_memory_utilization must be between zero and one.")
    if not str(engine.connector_module).strip():
        raise ValueError("engine.connector_module must be an importable module path.")
    kv = config.kv
    if kv.max_position < 1:
        raise ValueError("kv.max_position must be >= 1.")
    if kv.staging_slots < 1:
        raise ValueError("kv.staging_slots must be >= 1.")
    if kv.store_backend not in KNOWN_KV_STORE_BACKENDS:
        raise ValueError(
            f"kv.store_backend must be one of {KNOWN_KV_STORE_BACKENDS}, got {kv.store_backend!r}."
        )
    if kv.dtype.lower() not in KV_DTYPES:
        raise ValueError(f"kv.dtype must be one of {KV_DTYPES}, got {kv.dtype!r}.")
    index = config.index
    if index.n_neighbors < 1:
        raise ValueError("index.n_neighbors must be >= 1.")
    if index.beam_width < 1:
        raise ValueError("index.beam_width must be >= 1.")
    if max(index.beam_width, config.top_k) > MAX_JASPER_BEAM_WIDTH:
        raise ValueError(
            "Effective Jasper beam width must be <= "
            f"{MAX_JASPER_BEAM_WIDTH}; got max({index.beam_width}, {config.top_k})."
        )
    if config.embedding.batch_size < 1:
        raise ValueError("embedding.batch_size must be >= 1.")
    if not config.embedding.model:
        raise ValueError("embedding.model must be non-empty.")
    if not config.prompt.system_prompt:
        raise ValueError("prompt.system_prompt must be non-empty.")
    if not config.prompt.empty_context_text:
        raise ValueError("prompt.empty_context_text must be non-empty.")
    if config.generation.max_tokens < 0:
        raise ValueError("generation.max_tokens must be >= 0.")
