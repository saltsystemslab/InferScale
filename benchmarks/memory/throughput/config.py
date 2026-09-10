from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchmarks.common.config import (
    DEFAULT_EXTRACTION_LLM_BASE_URL,
    ConfigError,
    RuntimeConfig,
    embedding_api_key,
    expand_path,
    inferscale_section,
    int_list,
    load_json_object,
    load_runtime_config,
    optional,
    reject_unknown_keys,
    require,
    section_of,
    str_list,
)
from inferscale.v1.config import EngineConfig, KV_DTYPES
from inferscale.v1.index.jasper import MAX_JASPER_BEAM_WIDTH
from inferscale.v1.kv.connector_utils import (
    DEFAULT_KV_STAGING_SLOTS,
    DEFAULT_KV_STORE_BACKEND,
    KNOWN_KV_STORE_BACKENDS,
)

ALL_CONDITIONS = (
    "mem0_qdrant",
    "mem0_jasper",
    "kv_injection",
)

# Which vector backend a condition retrieves with. kv_injection performs the
# same Jasper top-k search as mem0_jasper and injects the retrieved chunks'
# KV instead of their text.
CONDITION_VECTOR_BACKENDS: dict[str, str | None] = {
    "mem0_qdrant": "qdrant",
    "mem0_jasper": "jasper",
    "kv_injection": "jasper",
}


def condition_vector_backend(condition: str) -> str | None:
    return CONDITION_VECTOR_BACKENDS[condition]


DEFAULT_USER_COUNTS = (50, 100, 150, 200)


@dataclass(slots=True)
class ThroughputConfig:
    model: str
    model_label: str
    results_dir: Path
    run_id: str
    dataset_path: Path = Path("data/locomo10.json")
    conditions: tuple[str, ...] = ALL_CONDITIONS
    user_counts: tuple[int, ...] = DEFAULT_USER_COUNTS
    requests_per_user: int = 2
    max_output_tokens: int = 50
    warmup_batches: int = 2
    top_k: int = 50
    context_window: int = 50
    seed: int = 42
    log_level: str = "INFO"
    kv_gpu_memory_utilization: float = 0.30
    kv_max_model_len: int = 32768
    kv_max_position: int = 32768
    kv_dtype: str = "bfloat16"
    kv_device: str = "cuda:0"
    kv_block_size: int = 16
    kv_connector_module: str = "inferscale.v1.kv.connector"
    kv_enable_prefix_caching: bool = True
    kv_store_backend: str = DEFAULT_KV_STORE_BACKEND
    kv_staging_slots: int = DEFAULT_KV_STAGING_SLOTS
    kv_chunk_cache_enabled: bool = True
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = None  # type: ignore[assignment]
    memory_llm_provider: str = "vllm"
    # Mem0 fact extraction always uses the answer model; see __post_init__.
    memory_llm_model: str | None = None
    memory_llm_base_url: str | None = DEFAULT_EXTRACTION_LLM_BASE_URL
    memory_llm_cache_dir: Path = None  # type: ignore[assignment]
    local_store_dir: Path = None  # type: ignore[assignment]
    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    def __post_init__(self) -> None:
        if any(
            value is None
            for value in (self.embedding_cache_dir, self.memory_llm_cache_dir, self.local_store_dir)
        ):
            layout = load_runtime_config().layout
            if self.embedding_cache_dir is None:
                self.embedding_cache_dir = layout.embedding_cache_dir
            if self.memory_llm_cache_dir is None:
                self.memory_llm_cache_dir = layout.memory_llm_cache_dir
            if self.local_store_dir is None:
                self.local_store_dir = layout.local_store_dir
        if self.memory_llm_model is None:
            self.memory_llm_model = self.model
        elif self.memory_llm_model != self.model:
            raise ValueError(
                "Mem0 fact extraction always uses the answer model; "
                f"memory_llm_model={self.memory_llm_model!r} conflicts with model={self.model!r}."
            )
        _validate_config(self)

    @property
    def run_dir(self) -> Path:
        return self.results_dir / "throughput" / self.run_id

    @property
    def local_store_scratch_dir(self) -> Path:
        return self.local_store_dir / "inferscale-bench-stores" / self.run_id

    def engine_config(self) -> EngineConfig:
        return EngineConfig(
            gpu_memory_utilization=self.kv_gpu_memory_utilization,
            max_model_len=self.kv_max_model_len,
            block_size=self.kv_block_size,
            enable_prefix_caching=self.kv_enable_prefix_caching,
            connector_module=self.kv_connector_module,
        )

    def to_jsonable(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        data = asdict(self)
        for key in ("results_dir", "dataset_path", "embedding_cache_dir", "memory_llm_cache_dir", "local_store_dir"):
            data[key] = str(data[key])
        data["user_counts"] = list(self.user_counts)
        data["conditions"] = list(self.conditions)
        if redact_secrets and data.get("embedding_api_key"):
            data["embedding_api_key"] = "<redacted>"
        return data

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ThroughputConfig":
        """Restore an internal, flattened worker snapshot, not an authored run file."""
        data = load_json_object(path)
        # Old snapshots cannot disable the now mandatory GPU text-to-KV map.
        data.pop("jasper_device_kv_selection", None)
        reject_unknown_keys(data, tuple(item.name for item in fields(cls)), "throughput worker snapshot")
        for key in ("model", "model_label", "run_id", "results_dir"):
            require(data, key, str, "throughput worker snapshot")
        for key in ("results_dir", "dataset_path", "embedding_cache_dir", "memory_llm_cache_dir", "local_store_dir"):
            if data.get(key) is not None:
                data[key] = Path(require(data, key, str, "throughput worker snapshot"))
        data["conditions"] = str_list(data, "conditions", "throughput worker snapshot")
        data["user_counts"] = int_list(data, "user_counts", "throughput worker snapshot")
        if data.get("embedding_api_key") == "<redacted>":
            data["embedding_api_key"] = (
                os.environ.get("LOCOMO_THROUGHPUT_EMBEDDING_API_KEY")
                or embedding_api_key()
            )
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, runtime: RuntimeConfig) -> "ThroughputConfig":
        """Build a run from authored JSON or a materialized JSON sweep cell."""
        return _build_run_config(data, runtime)


def user_counts_text(counts: Iterable[int]) -> str:
    return ",".join(str(count) for count in counts)


def default_run_id(model_label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_slug(model_label)}-{stamp}"


_RUN_KEYS = (
    "benchmark", "model", "dataset_path", "results_dir", "run_id", "conditions",
    "user_counts", "requests_per_user", "max_output_tokens", "warmup_batches",
    "top_k", "context_window", "seed", "log_level", "embedding_cache_enabled",
    "kv_chunk_cache_enabled", "inferscale", "mem0", "sweeps",
)


def load_throughput_config(path: str | Path, runtime: RuntimeConfig) -> ThroughputConfig:
    """Load benchmark settings from JSON and storage/model defaults from runtime JSON."""
    return ThroughputConfig.from_dict(load_json_object(path), runtime=runtime)


def _build_run_config(data: Mapping[str, Any], runtime: RuntimeConfig) -> ThroughputConfig:
    where = "memory-throughput"
    reject_unknown_keys(data, _RUN_KEYS, where)
    benchmark = optional(data, "benchmark", str, where, where)
    if benchmark != where:
        raise ConfigError(f"Expected benchmark {where!r}, got {benchmark!r}.")
    raw_model = require(data, "model", str, where).strip()
    if not raw_model:
        raise ConfigError(f"{where}.model must be non-empty.")
    top_k = optional(data, "top_k", int, 50, where)
    context_window = optional(data, "context_window", int, 50, where)
    max_output_tokens = optional(data, "max_output_tokens", int, 50, where)
    library = inferscale_section(
        data, model=runtime.resolve_model(raw_model), top_k=top_k,
        context_window=context_window, where=where,
    )
    nested = section_of(data, "inferscale", where)
    if "prompt" in nested:
        raise ConfigError(f"{where}.inferscale.prompt is fixed by the benchmark protocol.")
    embedding = section_of(nested, "embedding", f"{where}.inferscale", required=False)
    if "batch_size" in embedding:
        raise ConfigError(f"{where}.inferscale.embedding.batch_size is not supported; Mem0 replays individual facts.")
    generation = section_of(nested, "generation", f"{where}.inferscale", required=False)
    for key, expected in (("max_tokens", max_output_tokens), ("temperature", 0.0), ("top_p", 1.0)):
        if key in generation and generation[key] != expected:
            raise ConfigError(
                f"{where}.inferscale.generation.{key} must be {expected!r}; "
                "throughput uses fixed-length deterministic generation configured by max_output_tokens."
            )
    mem0 = section_of(data, "mem0", where, required=False)
    reject_unknown_keys(mem0, ("llm_base_url",), f"{where}.mem0")
    memory_llm_base_url = optional(
        mem0, "llm_base_url", str, DEFAULT_EXTRACTION_LLM_BASE_URL, f"{where}.mem0"
    )
    _validate_sweeps(section_of(data, "sweeps", where, required=False))
    label = runtime.model_label(raw_model)
    layout = runtime.layout
    config = ThroughputConfig(
        model=library.model,
        model_label=label,
        results_dir=expand_path(optional(data, "results_dir", str, str(layout.results_root), where), root=runtime.root),
        run_id=optional(data, "run_id", str, None, where) or default_run_id(label),
        dataset_path=expand_path(optional(data, "dataset_path", str, "data/locomo10.json", where), root=runtime.root),
        conditions=str_list(data, "conditions", where, default=ALL_CONDITIONS),
        user_counts=int_list(data, "user_counts", where, default=DEFAULT_USER_COUNTS),
        requests_per_user=optional(data, "requests_per_user", int, 2, where),
        max_output_tokens=max_output_tokens,
        warmup_batches=optional(data, "warmup_batches", int, 2, where),
        top_k=top_k,
        context_window=library.context_window,
        seed=optional(data, "seed", int, 42, where),
        log_level=optional(data, "log_level", str, "INFO", where),
        kv_gpu_memory_utilization=library.engine.gpu_memory_utilization,
        kv_max_model_len=library.engine.max_model_len,
        kv_max_position=library.kv.max_position,
        kv_dtype=library.kv.dtype,
        kv_device=library.kv.device,
        kv_block_size=library.engine.block_size,
        kv_connector_module=library.engine.connector_module,
        kv_enable_prefix_caching=library.engine.enable_prefix_caching,
        kv_store_backend=library.kv.store_backend,
        kv_staging_slots=library.kv.staging_slots,
        kv_chunk_cache_enabled=optional(data, "kv_chunk_cache_enabled", bool, True, where),
        embedding_model=library.embedding.model,
        embedding_api_key=embedding_api_key(),
        embedding_base_url=library.embedding.base_url,
        embedding_cache_enabled=optional(data, "embedding_cache_enabled", bool, True, where),
        embedding_cache_dir=layout.embedding_cache_dir,
        memory_llm_base_url=memory_llm_base_url,
        memory_llm_cache_dir=layout.memory_llm_cache_dir,
        local_store_dir=layout.local_store_dir,
        jasper_n_neighbors=library.index.n_neighbors,
        jasper_alpha=library.index.alpha,
        jasper_workspace_budget=library.index.workspace_budget,
        jasper_beam_width=library.index.beam_width,
    )
    return config


def _validate_sweeps(sweeps: dict[str, Any]) -> None:
    for name in sweeps:
        where = f"memory-throughput.sweeps.{name}"
        sweep = section_of(sweeps, name, "memory-throughput.sweeps")
        reject_unknown_keys(sweep, ("run_prefix", "conditions", "kv_store_backend", "kv_staging_slots"), where)
        if not optional(sweep, "run_prefix", str, "throughput", where).strip():
            raise ConfigError(f"{where}.run_prefix must be non-empty.")
        conditions = str_list(sweep, "conditions", where, default=ALL_CONDITIONS)
        _validate_conditions(conditions, where)
        backend = optional(sweep, "kv_store_backend", str, DEFAULT_KV_STORE_BACKEND, where)
        if backend not in KNOWN_KV_STORE_BACKENDS:
            raise ConfigError(f"{where}.kv_store_backend must be one of {KNOWN_KV_STORE_BACKENDS}.")
        if optional(sweep, "kv_staging_slots", int, DEFAULT_KV_STAGING_SLOTS, where) < 1:
            raise ConfigError(f"{where}.kv_staging_slots must be greater than zero.")


def _validate_conditions(conditions: tuple[str, ...], where: str) -> None:
    if not conditions or any(condition not in ALL_CONDITIONS for condition in conditions):
        raise ConfigError(f"{where}.conditions must be a non-empty list containing only {ALL_CONDITIONS}.")
    if len(set(conditions)) != len(conditions):
        raise ConfigError(f"{where}.conditions must not contain duplicates.")


def _validate_config(config: ThroughputConfig) -> None:
    data = asdict(config)
    where = "memory-throughput"
    for name in (
        "requests_per_user", "max_output_tokens", "warmup_batches", "top_k", "context_window", "seed",
        "kv_max_model_len", "kv_max_position", "kv_block_size", "kv_staging_slots",
        "jasper_n_neighbors", "jasper_beam_width",
    ):
        value = require(data, name, int, where)
        lower_bound = 0 if name in {"warmup_batches", "context_window", "seed"} else 1
        if value < lower_bound:
            raise ConfigError(f"{where}.{name} must be at least {lower_bound}.")
    for name in ("kv_enable_prefix_caching", "kv_chunk_cache_enabled", "embedding_cache_enabled"):
        require(data, name, bool, where)
    for name in (
        "model", "model_label", "run_id", "log_level", "kv_dtype", "kv_device", "kv_connector_module",
        "kv_store_backend", "embedding_model", "memory_llm_provider", "memory_llm_model", "jasper_workspace_budget",
    ):
        if not require(data, name, str, where).strip():
            raise ConfigError(f"{where}.{name} must be non-empty.")
    for name in ("embedding_api_key", "embedding_base_url", "memory_llm_base_url"):
        if data[name] is not None:
            require(data, name, str, where)
    for name in ("kv_gpu_memory_utilization", "jasper_alpha"):
        value = require(data, name, (int, float), where)
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{where}.{name} must be a finite positive number.")
    if config.kv_gpu_memory_utilization >= 1:
        raise ConfigError(f"{where}.kv_gpu_memory_utilization must be between zero and one.")
    if config.kv_store_backend not in KNOWN_KV_STORE_BACKENDS:
        raise ConfigError(f"{where}.kv_store_backend must be one of {KNOWN_KV_STORE_BACKENDS}.")
    if config.kv_dtype.lower() not in KV_DTYPES:
        raise ConfigError(f"{where}.kv_dtype must be one of {KV_DTYPES}.")
    if config.memory_llm_provider != "vllm":
        raise ConfigError(f"{where}.memory_llm_provider must be vllm.")
    if config.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"{where}.log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
    _validate_conditions(config.conditions, where)
    if (
        not config.user_counts
        or any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in config.user_counts)
        or len(set(config.user_counts)) != len(config.user_counts)
    ):
        raise ConfigError(f"{where}.user_counts must be distinct positive integers.")
    uses_jasper = any(condition_vector_backend(condition) == "jasper" for condition in config.conditions)
    if uses_jasper and max(config.jasper_beam_width, config.top_k) > MAX_JASPER_BEAM_WIDTH:
        raise ConfigError(f"Effective Jasper beam width must be at most {MAX_JASPER_BEAM_WIDTH}.")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._").lower()
    return normalized or "model"
