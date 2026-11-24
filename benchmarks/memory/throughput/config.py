from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from benchmarks.common.config import (
    DEFAULT_EXTRACTION_LLM_BASE_URL,
    ConfigError,
    RuntimeConfig,
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
from benchmarks.common.lmcache.config import (
    ADAPTERS,
    DEFAULT_ADAPTER,
    DEFAULT_BLEND_CHECK_LAYERS,
    DEFAULT_BLEND_RECOMPUTE_RATIO,
    DEFAULT_BLEND_SPECIAL_STR,
    DEFAULT_CHUNK_SIZE,
    LMCacheConfig,
    parse_lmcache_config,
)
from inferscale.v1.config import KV_DTYPES, EngineConfig
from inferscale.v1.embedding.local import DEFAULT_EMBEDDING_MODEL
from inferscale.v1.index.jasper import MAX_JASPER_BEAM_WIDTH
from inferscale.v1.kv.connector_utils import (
    DEFAULT_KV_STAGING_SLOTS,
    DEFAULT_KV_STORE_BACKEND,
    KNOWN_KV_STORE_BACKENDS,
)

BENCHMARK_NAME = "memory-throughput"
LMCACHE_CONDITION = "lmcache_blend"
ALL_CONDITIONS = ("mem0_qdrant", "mem0_jasper", LMCACHE_CONDITION, "kv_injection")
DEFAULT_CONDITIONS = tuple(
    condition for condition in ALL_CONDITIONS if condition != LMCACHE_CONDITION
)
CONDITION_VECTOR_BACKENDS = {
    "mem0_qdrant": "qdrant",
    "mem0_jasper": "jasper",
    LMCACHE_CONDITION: "qdrant",
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
    "lmcache",
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
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS
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
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_cache_enabled: bool = True
    memory_llm_provider: str = "vllm"
    memory_llm_model: str | None = None
    memory_llm_base_url: str | None = DEFAULT_EXTRACTION_LLM_BASE_URL
    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64
    lmcache_max_local_cpu_size_gb: float | None = None
    lmcache_chunk_size: int = DEFAULT_CHUNK_SIZE
    lmcache_blend_recompute_ratio: float = DEFAULT_BLEND_RECOMPUTE_RATIO
    lmcache_blend_check_layers: tuple[int, ...] = DEFAULT_BLEND_CHECK_LAYERS
    lmcache_blend_special_str: str = DEFAULT_BLEND_SPECIAL_STR
    lmcache_adapter: str = DEFAULT_ADAPTER

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
        settings = asdict(self)
        for name in PATH_FIELDS:
            settings[name] = str(settings[name])
        settings["conditions"] = list(self.conditions)
        settings["user_counts"] = list(self.user_counts)
        settings["lmcache_blend_check_layers"] = list(self.lmcache_blend_check_layers)
        return settings

    @classmethod
    def from_json_file(cls, path: str | Path) -> ThroughputConfig:
        where = "throughput worker snapshot"
        snapshot = load_json_object(path)
        reject_unknown_keys(snapshot, FIELD_NAMES, where)
        for name in ("model", "model_label", "run_id"):
            require(snapshot, name, str, where)
        for name in PATH_FIELDS:
            snapshot[name] = Path(require(snapshot, name, str, where))
        snapshot["conditions"] = str_list(snapshot, "conditions", where)
        snapshot["user_counts"] = int_list(snapshot, "user_counts", where)
        snapshot["lmcache_blend_check_layers"] = int_list(
            snapshot, "lmcache_blend_check_layers", where
        )
        return cls(**snapshot)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, runtime: RuntimeConfig) -> ThroughputConfig:
        where = BENCHMARK_NAME
        reject_unknown_keys(data, RUN_KEYS, where)
        benchmark = optional(data, "benchmark", str, BENCHMARK_NAME, where)
        if benchmark != BENCHMARK_NAME:
            raise ConfigError(f"Expected benchmark {BENCHMARK_NAME!r}, got {benchmark!r}.")
        model_name = require(data, "model", str, where).strip()
        if not model_name:
            raise ConfigError(f"{where}.model must be non-empty.")
        run = _run_settings(data, where)
        library = _library_config(data, runtime, model_name, run, where)
        sweeps = section_of(data, "sweeps", where, required=False)
        _validate_sweeps(sweeps)
        lmcache = _lmcache_settings(data, run["conditions"], sweeps, where)
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
            memory_llm_base_url=_extraction_endpoint(data, where),
            **run,
            **_library_fields(library),
            **_lmcache_fields(lmcache),
        )


FIELD_NAMES = tuple(field.name for field in fields(ThroughputConfig))


def condition_vector_backend(condition: str) -> str:
    return CONDITION_VECTOR_BACKENDS[condition]


def lmcache_config(config: ThroughputConfig) -> LMCacheConfig:
    if config.lmcache_max_local_cpu_size_gb is None:
        raise ConfigError(
            f"{BENCHMARK_NAME}.lmcache.max_local_cpu_size_gb is required for the "
            f"{LMCACHE_CONDITION} condition."
        )
    return LMCacheConfig(
        max_local_cpu_size_gb=config.lmcache_max_local_cpu_size_gb,
        chunk_size=config.lmcache_chunk_size,
        blend_recompute_ratio=config.lmcache_blend_recompute_ratio,
        blend_check_layers=tuple(config.lmcache_blend_check_layers),
        blend_special_str=config.lmcache_blend_special_str,
        adapter=config.lmcache_adapter,
    )


def jasper_effective_beam_width(config: ThroughputConfig) -> int:
    return max(config.jasper_beam_width, config.top_k)


def user_counts_text(counts: Iterable[int]) -> str:
    return ",".join(str(count) for count in counts)


def default_run_id(model_label: str) -> str:
    return f"{slug(model_label)}-{stamp_now()}"


def load_throughput_config(path: str | Path, runtime: RuntimeConfig) -> ThroughputConfig:
    return ThroughputConfig.from_dict(load_json_object(path), runtime=runtime)


def _run_settings(data: Mapping[str, Any], where: str) -> dict[str, Any]:
    return {
        "conditions": str_list(data, "conditions", where, default=DEFAULT_CONDITIONS),
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
    library = inferscale_section(
        data,
        model=runtime.resolve_model(model_name),
        top_k=run["top_k"],
        context_window=run["context_window"],
        where=where,
    )
    section = section_of(data, "inferscale", where)
    section_where = f"{where}.inferscale"
    if "prompt" in section:
        raise ConfigError(f"{section_where}.prompt is fixed by the benchmark protocol.")
    embedding = section_of(section, "embedding", section_where, required=False)
    if "batch_size" in embedding:
        raise ConfigError(
            f"{section_where}.embedding.batch_size is not supported; Mem0 replays individual facts."
        )
    generation = section_of(section, "generation", section_where, required=False)
    fixed_generation = {
        "max_tokens": run["max_output_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
    }
    for key, expected in fixed_generation.items():
        if key in generation and generation[key] != expected:
            raise ConfigError(
                f"{section_where}.generation.{key} must be {expected!r}; throughput uses "
                "fixed-length deterministic generation configured by max_output_tokens."
            )
    return library


def _library_fields(library: Any) -> dict[str, Any]:
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
        "jasper_n_neighbors": library.index.n_neighbors,
        "jasper_alpha": library.index.alpha,
        "jasper_workspace_budget": library.index.workspace_budget,
        "jasper_beam_width": library.index.beam_width,
    }


def _lmcache_settings(
    data: Mapping[str, Any], conditions: tuple[str, ...], sweeps: Mapping[str, Any], where: str
) -> LMCacheConfig | None:
    if data.get("lmcache") is not None:
        return parse_lmcache_config(section_of(data, "lmcache", where), f"{where}.lmcache")
    sweep_conditions = [
        condition for sweep in sweeps.values() for condition in sweep.get("conditions") or ()
    ]
    if LMCACHE_CONDITION in conditions or LMCACHE_CONDITION in sweep_conditions:
        raise ConfigError(f"{where}.lmcache is required when {LMCACHE_CONDITION} is a condition.")
    return None


def _lmcache_fields(lmcache: LMCacheConfig | None) -> dict[str, Any]:
    if lmcache is None:
        return {}
    return {
        "lmcache_max_local_cpu_size_gb": lmcache.max_local_cpu_size_gb,
        "lmcache_chunk_size": lmcache.chunk_size,
        "lmcache_blend_recompute_ratio": lmcache.blend_recompute_ratio,
        "lmcache_blend_check_layers": tuple(lmcache.blend_check_layers),
        "lmcache_blend_special_str": lmcache.blend_special_str,
        "lmcache_adapter": lmcache.adapter,
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
        _validate_conditions(
            str_list(sweep, "conditions", where, default=DEFAULT_CONDITIONS), where
        )
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
    "lmcache_chunk_size",
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
    "lmcache_blend_special_str",
    "lmcache_adapter",
)
_OPTIONAL_TEXT_FIELDS = ("memory_llm_base_url",)
_POSITIVE_NUMBER_FIELDS = ("kv_gpu_memory_utilization", "jasper_alpha")


def _validate_config(config: ThroughputConfig) -> None:
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
    _validate_lmcache_fields(config, where)
    uses_jasper = any(
        condition_vector_backend(condition) == "jasper" for condition in config.conditions
    )
    if uses_jasper and jasper_effective_beam_width(config) > MAX_JASPER_BEAM_WIDTH:
        raise ConfigError(f"Effective Jasper beam width must be at most {MAX_JASPER_BEAM_WIDTH}.")


def _validate_field_types(settings: Mapping[str, Any], where: str) -> None:
    for name in _POSITIVE_INT_FIELDS:
        if require(settings, name, int, where) < 1:
            raise ConfigError(f"{where}.{name} must be at least 1.")
    for name in _NON_NEGATIVE_INT_FIELDS:
        if require(settings, name, int, where) < 0:
            raise ConfigError(f"{where}.{name} must be at least 0.")
    for name in _BOOL_FIELDS:
        require(settings, name, bool, where)
    for name in _TEXT_FIELDS:
        if not require(settings, name, str, where).strip():
            raise ConfigError(f"{where}.{name} must be non-empty.")
    for name in _OPTIONAL_TEXT_FIELDS:
        if settings[name] is not None:
            require(settings, name, str, where)
    for name in _POSITIVE_NUMBER_FIELDS:
        number = require(settings, name, (int, float), where)
        if not math.isfinite(number) or number <= 0:
            raise ConfigError(f"{where}.{name} must be a finite positive number.")


def _validate_lmcache_fields(config: ThroughputConfig, where: str) -> None:
    size = config.lmcache_max_local_cpu_size_gb
    if size is not None and not (_is_number(size) and math.isfinite(size) and size > 0):
        raise ConfigError(
            f"{where}.lmcache_max_local_cpu_size_gb must be a finite positive number."
        )
    ratio = config.lmcache_blend_recompute_ratio
    if not (_is_number(ratio) and math.isfinite(ratio) and 0 <= ratio <= 1):
        raise ConfigError(f"{where}.lmcache_blend_recompute_ratio must be between 0 and 1.")
    layers = config.lmcache_blend_check_layers
    distinct_layers = (
        isinstance(layers, tuple)
        and bool(layers)
        and all(_is_index(layer) for layer in layers)
        and len(set(layers)) == len(layers)
    )
    if not distinct_layers:
        raise ConfigError(
            f"{where}.lmcache_blend_check_layers must be distinct non-negative integers."
        )
    if config.lmcache_adapter not in ADAPTERS:
        raise ConfigError(f"{where}.lmcache_adapter must be one of {ADAPTERS}.")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_user_counts(user_counts: tuple[int, ...], where: str) -> None:
    all_positive_integers = all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 1
        for count in user_counts
    )
    if not user_counts or not all_positive_integers or len(set(user_counts)) != len(user_counts):
        raise ConfigError(f"{where}.user_counts must be distinct positive integers.")
