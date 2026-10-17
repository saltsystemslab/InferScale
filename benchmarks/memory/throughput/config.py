"""Settings of one throughput run, flattened from the authored JSON and the runtime layout.

A run is authored as a JSON file under configs/memory/throughput whose
``inferscale`` section is the library configuration. The runner writes the
flattened settings to ``config.json`` in the run directory, and every worker
subprocess restores them from there so each measurement uses exactly the
settings of the run.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from benchmarks.common.config import (
    DEFAULT_EXTRACTION_LLM_BASE_URL,
    ConfigError,
    RuntimeConfig,
    embedding_api_key,
    load_json_object,
    slug,
    stamp_now,
)
from benchmarks.common.inferscale_config import inferscale_section
from benchmarks.common.json_fields import (
    expand_path,
    int_list,
    optional,
    reject_unknown_keys,
    require,
    section_of,
    str_list,
)
from inferscale.v1.config import KV_DTYPES, EngineConfig
from inferscale.v1.index.jasper import MAX_JASPER_BEAM_WIDTH
from inferscale.v1.kv.connector_utils import (
    DEFAULT_KV_STAGING_SLOTS,
    DEFAULT_KV_STORE_BACKEND,
    KNOWN_KV_STORE_BACKENDS,
)

BENCHMARK_NAME = "memory-throughput"
ALL_CONDITIONS = ("mem0_qdrant", "mem0_jasper", "kv_injection")
# Which vector backend a condition retrieves with. kv_injection performs the
# same Jasper top-k search as mem0_jasper and injects the retrieved chunks'
# KV instead of their text.
CONDITION_VECTOR_BACKENDS = {
    "mem0_qdrant": "qdrant",
    "mem0_jasper": "jasper",
    "kv_injection": "jasper",
}
DEFAULT_USER_COUNTS = (50, 100, 150, 200)
DEFAULT_DATASET_PATH = "data/locomo10.json"
DEFAULT_REQUESTS_PER_USER = 2
DEFAULT_MAX_OUTPUT_TOKENS = 50
DEFAULT_WARMUP_BATCHES = 2
DEFAULT_TOP_K = 50
DEFAULT_CONTEXT_WINDOW = 50
DEFAULT_SEED = 42
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
REDACTED_API_KEY = "<redacted>"
# The runner hands the embedding key to its workers through the environment,
# never through the snapshot on disk.
WORKER_API_KEY_VARIABLE = "LOCOMO_THROUGHPUT_EMBEDDING_API_KEY"
PATH_FIELDS = (
    "results_dir",
    "cache_root",
    "embedding_cache_dir",
    "memory_llm_cache_dir",
    "local_store_dir",
    "dataset_path",
)
RUN_KEYS = (
    "benchmark",
    "model",
    "dataset_path",
    "results_dir",
    "run_id",
    "conditions",
    "user_counts",
    "requests_per_user",
    "max_output_tokens",
    "warmup_batches",
    "top_k",
    "context_window",
    "seed",
    "log_level",
    "embedding_cache_enabled",
    "kv_chunk_cache_enabled",
    "inferscale",
    "mem0",
    "sweeps",
)
SWEEP_KEYS = ("run_prefix", "conditions", "kv_store_backend", "kv_staging_slots")


@dataclass(slots=True)
class ThroughputConfig:
    model: str
    model_label: str
    results_dir: Path
    run_id: str
    cache_root: Path
    embedding_cache_dir: Path
    memory_llm_cache_dir: Path
    local_store_dir: Path
    dataset_path: Path = Path(DEFAULT_DATASET_PATH)
    conditions: tuple[str, ...] = ALL_CONDITIONS
    user_counts: tuple[int, ...] = DEFAULT_USER_COUNTS
    requests_per_user: int = DEFAULT_REQUESTS_PER_USER
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    warmup_batches: int = DEFAULT_WARMUP_BATCHES
    top_k: int = DEFAULT_TOP_K
    context_window: int = DEFAULT_CONTEXT_WINDOW
    seed: int = DEFAULT_SEED
    log_level: str = DEFAULT_LOG_LEVEL
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
    memory_llm_provider: str = "vllm"
    # Mem0 fact extraction always uses the answer model; None resolves to it.
    memory_llm_model: str | None = None
    memory_llm_base_url: str | None = DEFAULT_EXTRACTION_LLM_BASE_URL
    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    def __post_init__(self) -> None:
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

    def to_jsonable(self) -> dict[str, Any]:
        """The snapshot written to the run directory; the API key is never written out."""
        data = asdict(self)
        for name in PATH_FIELDS:
            data[name] = str(data[name])
        data["conditions"] = list(self.conditions)
        data["user_counts"] = list(self.user_counts)
        if data["embedding_api_key"]:
            data["embedding_api_key"] = REDACTED_API_KEY
        return data

    @classmethod
    def from_json_file(cls, path: str | Path) -> ThroughputConfig:
        """Restore the flattened snapshot the runner wrote, not an authored run file."""
        where = "throughput worker snapshot"
        data = load_json_object(path)
        reject_unknown_keys(data, FIELD_NAMES, where)
        for name in ("model", "model_label", "run_id"):
            require(data, name, str, where)
        for name in PATH_FIELDS:
            data[name] = Path(require(data, name, str, where))
        data["conditions"] = str_list(data, "conditions", where)
        data["user_counts"] = int_list(data, "user_counts", where)
        if data.get("embedding_api_key") == REDACTED_API_KEY:
            data["embedding_api_key"] = (
                os.environ.get(WORKER_API_KEY_VARIABLE) or embedding_api_key()
            )
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, runtime: RuntimeConfig) -> ThroughputConfig:
        """Build a run from authored JSON or a materialized sweep cell."""
        where = BENCHMARK_NAME
        reject_unknown_keys(data, RUN_KEYS, where)
        benchmark = optional(data, "benchmark", str, where, where)
        if benchmark != where:
            raise ConfigError(f"Expected benchmark {where!r}, got {benchmark!r}.")
        model_name = require(data, "model", str, where).strip()
        if not model_name:
            raise ConfigError(f"{where}.model must be non-empty.")
        run = _run_settings(data, where)
        library = _library_config(data, runtime, model_name, run, where)
        _validate_sweeps(section_of(data, "sweeps", where, required=False))
        label = runtime.model_label(model_name)
        layout = runtime.layout
        results_dir = optional(data, "results_dir", str, str(layout.results_root), where)
        dataset_path = optional(data, "dataset_path", str, DEFAULT_DATASET_PATH, where)
        return cls(
            model=library.model,
            model_label=label,
            results_dir=expand_path(results_dir, root=runtime.root),
            run_id=optional(data, "run_id", str, None, where) or default_run_id(label),
            cache_root=layout.cache_root,
            embedding_cache_dir=layout.embedding_cache_dir,
            memory_llm_cache_dir=layout.memory_llm_cache_dir,
            local_store_dir=layout.local_store_dir,
            dataset_path=expand_path(dataset_path, root=runtime.root),
            embedding_api_key=embedding_api_key(),
            memory_llm_base_url=_extraction_endpoint(data, where),
            **run,
            **_library_fields(library),
        )


FIELD_NAMES = tuple(item.name for item in fields(ThroughputConfig))


def condition_vector_backend(condition: str) -> str:
    return CONDITION_VECTOR_BACKENDS[condition]


def jasper_effective_beam_width(config: ThroughputConfig) -> int:
    """Jasper searches at least top_k wide so every retrieved fact is a candidate."""
    return max(config.jasper_beam_width, config.top_k)


def user_counts_text(counts: Iterable[int]) -> str:
    return ",".join(str(count) for count in counts)


def default_run_id(model_label: str) -> str:
    return f"{slug(model_label)}-{stamp_now()}"


def load_throughput_config(path: str | Path, runtime: RuntimeConfig) -> ThroughputConfig:
    """Load benchmark settings from JSON and storage and model defaults from the runtime JSON."""
    return ThroughputConfig.from_dict(load_json_object(path), runtime=runtime)


def _run_settings(data: Mapping[str, Any], where: str) -> dict[str, Any]:
    """The workload and cache switches authored at the top level of the run JSON."""
    return {
        "conditions": str_list(data, "conditions", where, default=ALL_CONDITIONS),
        "user_counts": int_list(data, "user_counts", where, default=DEFAULT_USER_COUNTS),
        "requests_per_user": optional(
            data, "requests_per_user", int, DEFAULT_REQUESTS_PER_USER, where
        ),
        "max_output_tokens": optional(
            data, "max_output_tokens", int, DEFAULT_MAX_OUTPUT_TOKENS, where
        ),
        "warmup_batches": optional(data, "warmup_batches", int, DEFAULT_WARMUP_BATCHES, where),
        "top_k": optional(data, "top_k", int, DEFAULT_TOP_K, where),
        "context_window": optional(data, "context_window", int, DEFAULT_CONTEXT_WINDOW, where),
        "seed": optional(data, "seed", int, DEFAULT_SEED, where),
        "log_level": optional(data, "log_level", str, DEFAULT_LOG_LEVEL, where),
        "embedding_cache_enabled": optional(data, "embedding_cache_enabled", bool, True, where),
        "kv_chunk_cache_enabled": optional(data, "kv_chunk_cache_enabled", bool, True, where),
    }


def _library_config(
    data: Mapping[str, Any],
    runtime: RuntimeConfig,
    model_name: str,
    run: Mapping[str, Any],
    where: str,
) -> Any:
    """The validated ``inferscale`` section, with the parts the protocol fixes rejected."""
    library = inferscale_section(
        data,
        model=runtime.resolve_model(model_name),
        top_k=run["top_k"],
        context_window=run["context_window"],
        where=where,
    )
    nested = section_of(data, "inferscale", where)
    nested_where = f"{where}.inferscale"
    if "prompt" in nested:
        raise ConfigError(f"{nested_where}.prompt is fixed by the benchmark protocol.")
    embedding = section_of(nested, "embedding", nested_where, required=False)
    if "batch_size" in embedding:
        raise ConfigError(
            f"{nested_where}.embedding.batch_size is not supported; Mem0 replays individual facts."
        )
    generation = section_of(nested, "generation", nested_where, required=False)
    fixed = {"max_tokens": run["max_output_tokens"], "temperature": 0.0, "top_p": 1.0}
    for key, expected in fixed.items():
        if key in generation and generation[key] != expected:
            raise ConfigError(
                f"{nested_where}.generation.{key} must be {expected!r}; throughput uses "
                "fixed-length deterministic generation configured by max_output_tokens."
            )
    return library


def _library_fields(library: Any) -> dict[str, Any]:
    """The flattened engine, KV, embedding, and index settings of the library section."""
    return {
        "kv_gpu_memory_utilization": library.engine.gpu_memory_utilization,
        "kv_max_model_len": library.engine.max_model_len,
        "kv_max_position": library.kv.max_position,
        "kv_dtype": library.kv.dtype,
        "kv_device": library.kv.device,
        "kv_block_size": library.engine.block_size,
        "kv_connector_module": library.engine.connector_module,
        "kv_enable_prefix_caching": library.engine.enable_prefix_caching,
        "kv_store_backend": library.kv.store_backend,
        "kv_staging_slots": library.kv.staging_slots,
        "embedding_model": library.embedding.model,
        "embedding_base_url": library.embedding.base_url,
        "jasper_n_neighbors": library.index.n_neighbors,
        "jasper_alpha": library.index.alpha,
        "jasper_workspace_budget": library.index.workspace_budget,
        "jasper_beam_width": library.index.beam_width,
    }


def _extraction_endpoint(data: Mapping[str, Any], where: str) -> str | None:
    mem0 = section_of(data, "mem0", where, required=False)
    mem0_where = f"{where}.mem0"
    reject_unknown_keys(mem0, ("llm_base_url",), mem0_where)
    return optional(mem0, "llm_base_url", str, DEFAULT_EXTRACTION_LLM_BASE_URL, mem0_where)


def _validate_sweeps(sweeps: Mapping[str, Any]) -> None:
    for name in sweeps:
        where = f"{BENCHMARK_NAME}.sweeps.{name}"
        sweep = section_of(sweeps, name, f"{BENCHMARK_NAME}.sweeps")
        reject_unknown_keys(sweep, SWEEP_KEYS, where)
        if not optional(sweep, "run_prefix", str, "throughput", where).strip():
            raise ConfigError(f"{where}.run_prefix must be non-empty.")
        _validate_conditions(str_list(sweep, "conditions", where, default=ALL_CONDITIONS), where)
        backend = optional(sweep, "kv_store_backend", str, DEFAULT_KV_STORE_BACKEND, where)
        if backend not in KNOWN_KV_STORE_BACKENDS:
            raise ConfigError(f"{where}.kv_store_backend must be one of {KNOWN_KV_STORE_BACKENDS}.")
        if optional(sweep, "kv_staging_slots", int, DEFAULT_KV_STAGING_SLOTS, where) < 1:
            raise ConfigError(f"{where}.kv_staging_slots must be greater than zero.")


def _validate_conditions(conditions: tuple[str, ...], where: str) -> None:
    if not conditions or any(condition not in ALL_CONDITIONS for condition in conditions):
        raise ConfigError(
            f"{where}.conditions must be a non-empty list containing only {ALL_CONDITIONS}."
        )
    if len(set(conditions)) != len(conditions):
        raise ConfigError(f"{where}.conditions must not contain duplicates.")


_POSITIVE_INT_FIELDS = (
    "requests_per_user",
    "max_output_tokens",
    "top_k",
    "kv_max_model_len",
    "kv_max_position",
    "kv_block_size",
    "kv_staging_slots",
    "jasper_n_neighbors",
    "jasper_beam_width",
)
_NON_NEGATIVE_INT_FIELDS = ("warmup_batches", "context_window", "seed")
_BOOL_FIELDS = ("kv_enable_prefix_caching", "kv_chunk_cache_enabled", "embedding_cache_enabled")
_TEXT_FIELDS = (
    "model",
    "model_label",
    "run_id",
    "log_level",
    "kv_dtype",
    "kv_device",
    "kv_connector_module",
    "kv_store_backend",
    "embedding_model",
    "memory_llm_provider",
    "memory_llm_model",
    "jasper_workspace_budget",
)
_OPTIONAL_TEXT_FIELDS = ("embedding_api_key", "embedding_base_url", "memory_llm_base_url")
_POSITIVE_NUMBER_FIELDS = ("kv_gpu_memory_utilization", "jasper_alpha")


def _validate_config(config: ThroughputConfig) -> None:
    """Check field types and ranges; snapshots restored from JSON arrive untyped."""
    where = BENCHMARK_NAME
    _validate_field_types(asdict(config), where)
    if config.kv_gpu_memory_utilization >= 1:
        raise ConfigError(f"{where}.kv_gpu_memory_utilization must be between zero and one.")
    if config.kv_store_backend not in KNOWN_KV_STORE_BACKENDS:
        raise ConfigError(f"{where}.kv_store_backend must be one of {KNOWN_KV_STORE_BACKENDS}.")
    if config.kv_dtype.lower() not in KV_DTYPES:
        raise ConfigError(f"{where}.kv_dtype must be one of {KV_DTYPES}.")
    if config.memory_llm_provider != "vllm":
        raise ConfigError(f"{where}.memory_llm_provider must be vllm.")
    if config.log_level.upper() not in LOG_LEVELS:
        raise ConfigError(f"{where}.log_level must be one of {', '.join(LOG_LEVELS)}.")
    _validate_conditions(config.conditions, where)
    _validate_user_counts(config.user_counts, where)
    uses_jasper = any(
        condition_vector_backend(condition) == "jasper" for condition in config.conditions
    )
    if uses_jasper and jasper_effective_beam_width(config) > MAX_JASPER_BEAM_WIDTH:
        raise ConfigError(f"Effective Jasper beam width must be at most {MAX_JASPER_BEAM_WIDTH}.")


def _validate_field_types(data: Mapping[str, Any], where: str) -> None:
    for name in _POSITIVE_INT_FIELDS:
        if require(data, name, int, where) < 1:
            raise ConfigError(f"{where}.{name} must be at least 1.")
    for name in _NON_NEGATIVE_INT_FIELDS:
        if require(data, name, int, where) < 0:
            raise ConfigError(f"{where}.{name} must be at least 0.")
    for name in _BOOL_FIELDS:
        require(data, name, bool, where)
    for name in _TEXT_FIELDS:
        if not require(data, name, str, where).strip():
            raise ConfigError(f"{where}.{name} must be non-empty.")
    for name in _OPTIONAL_TEXT_FIELDS:
        if data[name] is not None:
            require(data, name, str, where)
    for name in _POSITIVE_NUMBER_FIELDS:
        value = require(data, name, (int, float), where)
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{where}.{name} must be a finite positive number.")


def _validate_user_counts(user_counts: tuple[int, ...], where: str) -> None:
    valid = all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 1
        for count in user_counts
    )
    if not user_counts or not valid or len(set(user_counts)) != len(user_counts):
        raise ConfigError(f"{where}.user_counts must be distinct positive integers.")
