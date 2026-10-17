"""The JSON contract of one RAG benchmark run and its validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.common.config import (
    REDACTED,
    ConfigError,
    RuntimeConfig,
    embedding_api_key,
    load_json_object,
    load_runtime_config,
    stamp_now,
)
from benchmarks.common.inferscale_config import inferscale_section
from benchmarks.common.json_fields import (
    expand_path,
    int_list,
    optional,
    reject_unknown_keys,
    section_of,
    str_list,
)
from benchmarks.common.judge_config import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_PROVIDER,
    JudgeConfig,
    judge_config,
)
from benchmarks.rag.datasets import get_dataset
from inferscale.v1.config import EmbeddingConfig, EngineConfig
from inferscale.v1.index.jasper import MAX_JASPER_BEAM_WIDTH

WHERE = "rag"
ANSWER_BACKENDS = ("kv-injection", "prompt-injection")
# The corpus chunk KV is host-RAM resident at answer time, loaded once from
# the precompute cache; there is no GPU-resident corpus backend.
KV_STORE_BACKEND = "cpu"
STAGES = ("estimate", "preembed", "precompute-kv", "run", "judge")
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
RUN_KEYS = (
    "benchmark",
    "dataset",
    "data_dir",
    "models",
    "model",
    "results_dir",
    "run_id",
    "answer_backend",
    "chunk_size",
    "context_window",
    "top_k",
    "max_queries",
    "log_every",
    "log_level",
    "embed_batch_size",
    "embedding_cache_enabled",
    "embedding_cache_dir",
    "kv_chunk_cache_root",
    "skip_judge",
    "rejudge",
    "inferscale",
    "judge",
    "sweeps",
)
LIBRARY_SECTIONS = ("engine", "kv", "index", "embedding", "generation")
SWEEP_KEYS = ("top_k", "answer_backends")
PATH_KEYS = ("data_dir", "results_dir", "embedding_cache_dir", "kv_chunk_cache_root")
SECRET_KEYS = ("judge_api_key", "embedding_api_key")

DEFAULT_DATASET = "multihoprag"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CONTEXT_WINDOW = 5
DEFAULT_TOP_K = 15
DEFAULT_MAX_ANSWER_TOKENS = 64
DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_EMBEDDING_MODEL = EmbeddingConfig().model
# Parse-time slack for the scaffold and the templated question on top of the
# retrieved chunks; the answer path still enforces the exact budget per query.
MEMORY_BUDGET_MARGIN_TOKENS = 512


@dataclass(slots=True)
class RagBenchConfig:
    """One run's settings. ``from_dict`` fills them from JSON and the runtime; bare construction
    takes the plain defaults below and performs no I/O."""

    dataset_name: str = DEFAULT_DATASET
    data_dir: Path | None = None
    results_dir: Path = Path("results")
    run_id: str = field(default_factory=stamp_now)

    model: str = "llama"
    answer_backend: str = "kv-injection"

    chunk_size: int = DEFAULT_CHUNK_SIZE
    context_window: int = DEFAULT_CONTEXT_WINDOW
    top_k: int = DEFAULT_TOP_K

    judge_provider: str = DEFAULT_JUDGE_PROVIDER
    judge_model: str = ""
    judge_base_url: str | None = None
    judge_api_key: str | None = None

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path | None = None
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE

    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    temperature: float = 0.0
    top_p: float = 1.0
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
    max_judge_tokens: int = DEFAULT_JUDGE_MAX_TOKENS

    kv_connector_module: str = "inferscale.v1.kv.connector"
    kv_gpu_memory_utilization: float = 0.40
    kv_block_size: int = 16
    kv_max_model_len: int = 32768
    kv_max_position: int = 32768
    kv_dtype: str = "bfloat16"
    kv_device: str = "cuda:0"
    kv_enable_prefix_caching: bool = True
    kv_chunk_cache_root: Path | None = None

    max_queries: int | None = None
    log_every: int = 25
    log_level: str = "INFO"
    skip_judge: bool = False
    rejudge: bool = False

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = Path("data") / self.dataset_name

    def result_mode(self) -> str:
        return "rag-kv" if self.answer_backend == "kv-injection" else "rag-prompt-injection"

    @property
    def jasper_effective_beam_width(self) -> int:
        return max(self.jasper_beam_width, self.top_k)

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id

    def engine_config(self) -> EngineConfig:
        return EngineConfig(
            gpu_memory_utilization=self.kv_gpu_memory_utilization,
            max_model_len=self.kv_max_model_len,
            block_size=self.kv_block_size,
            enable_prefix_caching=self.kv_enable_prefix_caching,
            connector_module=self.kv_connector_module,
        )

    def judge(self) -> JudgeConfig:
        return JudgeConfig(
            provider=self.judge_provider,
            model=self.judge_model,
            base_url=self.judge_base_url,
            api_key=self.judge_api_key,
            max_tokens=self.max_judge_tokens,
        )

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        data["mode"] = self.result_mode()
        data["jasper_effective_beam_width"] = self.jasper_effective_beam_width
        data["kv_store_backend"] = KV_STORE_BACKEND
        for key in PATH_KEYS:
            if data[key] is not None:
                data[key] = str(data[key])
        for key in SECRET_KEYS:
            if data[key]:
                data[key] = REDACTED
        return data

    @classmethod
    def from_json_file(
        cls, path: str | Path, *, runtime: RuntimeConfig | None = None
    ) -> RagBenchConfig:
        return cls.from_dict(load_json_object(path), runtime=runtime)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, runtime: RuntimeConfig | None = None
    ) -> RagBenchConfig:
        runtime = runtime or load_runtime_config()
        reject_unknown_keys(data, RUN_KEYS, WHERE)
        if optional(data, "benchmark", str, "rag", WHERE) != "rag":
            raise ConfigError("rag.benchmark must be 'rag'.")
        model = _model(data, runtime)
        dataset_name = _dataset_name(data)
        top_k = optional(data, "top_k", int, DEFAULT_TOP_K, WHERE)
        context_window = optional(data, "context_window", int, DEFAULT_CONTEXT_WINDOW, WHERE)
        library = _library(data, model=model, top_k=top_k, context_window=context_window)
        judge = _judge(data, runtime)
        skip_judge = optional(data, "skip_judge", bool, False, WHERE) or judge.provider == "none"
        paths = _RunPaths.parse(data, runtime, dataset_name)
        config = cls(
            dataset_name=dataset_name,
            data_dir=paths.data_dir,
            results_dir=paths.results_dir,
            run_id=optional(data, "run_id", str, stamp_now(), WHERE),
            model=model,
            answer_backend=optional(data, "answer_backend", str, "kv-injection", WHERE),
            chunk_size=optional(data, "chunk_size", int, DEFAULT_CHUNK_SIZE, WHERE),
            context_window=library.context_window,
            top_k=top_k,
            judge_provider="none" if skip_judge else judge.provider,
            judge_model=judge.model,
            judge_base_url=judge.base_url,
            judge_api_key=judge.api_key,
            embedding_model=library.embedding.model,
            embedding_base_url=library.embedding.base_url,
            embedding_api_key=embedding_api_key(),
            embedding_cache_enabled=optional(data, "embedding_cache_enabled", bool, True, WHERE),
            embedding_cache_dir=paths.embedding_cache_dir,
            embed_batch_size=optional(
                data, "embed_batch_size", int, library.embedding.batch_size, WHERE
            ),
            jasper_n_neighbors=library.index.n_neighbors,
            jasper_alpha=library.index.alpha,
            jasper_workspace_budget=library.index.workspace_budget,
            jasper_beam_width=library.index.beam_width,
            temperature=library.generation.temperature,
            top_p=library.generation.top_p,
            max_answer_tokens=library.generation.max_tokens,
            max_judge_tokens=judge.max_tokens,
            kv_connector_module=library.engine.connector_module,
            kv_gpu_memory_utilization=library.engine.gpu_memory_utilization,
            kv_block_size=library.engine.block_size,
            kv_max_model_len=library.engine.max_model_len,
            kv_max_position=library.kv.max_position,
            kv_dtype=library.kv.dtype,
            kv_device=library.kv.device,
            kv_enable_prefix_caching=library.engine.enable_prefix_caching,
            kv_chunk_cache_root=paths.kv_chunk_cache_root,
            max_queries=optional(data, "max_queries", int, None, WHERE),
            log_every=optional(data, "log_every", int, 25, WHERE),
            log_level=optional(data, "log_level", str, "INFO", WHERE).upper(),
            skip_judge=skip_judge,
            rejudge=optional(data, "rejudge", bool, False, WHERE),
        )
        validate_rag_config(config)
        _validate_sweeps(section_of(data, "sweeps", WHERE, required=False))
        return config


@dataclass(slots=True, frozen=True)
class _RunPaths:
    """The run's directories: explicit in the JSON, else the runtime layout's defaults."""

    data_dir: Path
    results_dir: Path
    embedding_cache_dir: Path
    kv_chunk_cache_root: Path

    @classmethod
    def parse(cls, data: Mapping[str, Any], runtime: RuntimeConfig, dataset_name: str) -> _RunPaths:
        layout = runtime.layout
        return cls(
            data_dir=_path(data, "data_dir", runtime, runtime.root / "data" / dataset_name),
            results_dir=_path(data, "results_dir", runtime, layout.results_root),
            embedding_cache_dir=_path(
                data, "embedding_cache_dir", runtime, layout.embedding_cache_dir
            ),
            kv_chunk_cache_root=_path(
                data, "kv_chunk_cache_root", runtime, layout.rag_kv_chunk_cache_root
            ),
        )


def _path(data: Mapping[str, Any], key: str, runtime: RuntimeConfig, default: Path) -> Path:
    value = optional(data, key, str, None, WHERE)
    return expand_path(value, root=runtime.root) if value is not None else default


def _model(data: Mapping[str, Any], runtime: RuntimeConfig) -> str:
    if "model" in data and "models" in data:
        raise ConfigError("rag must specify model or models, not both.")
    if "models" in data:
        models = str_list(data, "models", WHERE)
    else:
        models = [optional(data, "model", str, "llama", WHERE)]
    if len(models) != 1:
        raise ConfigError(
            "A RAG invocation requires one model; use the sweep launcher for multiple models."
        )
    return runtime.resolve_model(models[0])


def _dataset_name(data: Mapping[str, Any]) -> str:
    name = optional(data, "dataset", str, DEFAULT_DATASET, WHERE)
    try:
        return get_dataset(name).name
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _library(data: Mapping[str, Any], *, model: str, top_k: int, context_window: int) -> Any:
    """The validated ``inferscale`` section with the RAG defaults filled in."""
    raw = dict(section_of(data, "inferscale", WHERE, required=False))
    # RAG caps answers shorter than the memory benchmark and keeps the corpus KV in host RAM.
    raw["generation"] = {
        "max_tokens": DEFAULT_MAX_ANSWER_TOKENS,
        **section_of(raw, "generation", "rag.inferscale", required=False),
    }
    raw["kv"] = {
        "store_backend": KV_STORE_BACKEND,
        **section_of(raw, "kv", "rag.inferscale", required=False),
    }
    library = inferscale_section(
        {"inferscale": raw}, model=model, top_k=top_k, context_window=context_window, where=WHERE
    )
    reject_unknown_keys(raw, LIBRARY_SECTIONS, "rag.inferscale")
    if library.kv.store_backend != KV_STORE_BACKEND:
        raise ConfigError("rag.inferscale.kv.store_backend must be cpu.")
    return library


def _judge(data: Mapping[str, Any], runtime: RuntimeConfig) -> JudgeConfig:
    judge = judge_config(
        section_of(data, "judge", WHERE, required=False), runtime, where="rag.judge"
    )
    if judge.with_evidence:
        raise ConfigError("rag.judge.with_evidence is not supported by the RAG judge.")
    return judge


def validate_rag_config(config: RagBenchConfig) -> None:
    _validate_choices(config)
    _validate_bounds(config)
    _validate_run_id(config.run_id)
    _validate_memory_budget(config)


def _validate_choices(config: RagBenchConfig) -> None:
    if config.answer_backend not in ANSWER_BACKENDS:
        raise ConfigError("rag.answer_backend must be kv-injection or prompt-injection.")
    if config.log_level not in LOG_LEVELS:
        raise ConfigError("rag.log_level is not a supported logging level.")
    if config.answer_backend == "prompt-injection" and not config.kv_enable_prefix_caching:
        raise ConfigError(
            "rag.answer_backend prompt-injection requires inferscale.engine.enable_prefix_caching."
        )


def _validate_bounds(config: RagBenchConfig) -> None:
    positive = {
        "top_k": config.top_k,
        "chunk_size": config.chunk_size,
        "embed_batch_size": config.embed_batch_size,
        "log_every": config.log_every,
        "max_answer_tokens": config.max_answer_tokens,
    }
    for name, value in positive.items():
        if value < 1:
            raise ConfigError(f"rag.{name} must be >= 1.")
    if config.context_window < 0:
        raise ConfigError("rag.context_window must be >= 0.")
    if config.max_queries is not None and config.max_queries < 1:
        raise ConfigError("rag.max_queries must be >= 1 or null.")


def _validate_run_id(run_id: str) -> None:
    if not run_id.strip() or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ConfigError("rag.run_id must be a non-empty directory name.")


def _validate_memory_budget(config: RagBenchConfig) -> None:
    budget = min(config.kv_max_position, config.kv_max_model_len - config.max_answer_tokens)
    if config.top_k * config.chunk_size + MEMORY_BUDGET_MARGIN_TOKENS > budget:
        raise ConfigError(
            f"top_k x chunk_size + {MEMORY_BUDGET_MARGIN_TOKENS} scaffold/query margin exceeds "
            f"the memory budget min(inferscale.kv.max_position, "
            f"inferscale.engine.max_model_len - inferscale.generation.max_tokens) = {budget}."
        )


def _validate_sweeps(sweeps: Mapping[str, Any]) -> None:
    for name in sweeps:
        where = f"rag.sweeps.{name}"
        sweep = section_of(sweeps, name, "rag.sweeps")
        reject_unknown_keys(sweep, SWEEP_KEYS, where)
        top_ks = int_list(sweep, "top_k", where, default=[DEFAULT_TOP_K])
        if any(k < 1 or k > MAX_JASPER_BEAM_WIDTH for k in top_ks):
            raise ConfigError(f"{where}.top_k must be between 1 and {MAX_JASPER_BEAM_WIDTH}.")
        backends = str_list(sweep, "answer_backends", where, default=list(ANSWER_BACKENDS))
        if any(backend not in ANSWER_BACKENDS for backend in backends):
            raise ConfigError(
                f"{where}.answer_backends must contain only kv-injection or prompt-injection."
            )


def load_rag_config(
    path: str | Path, *, runtime: RuntimeConfig | None = None, stage: str = "run"
) -> RagBenchConfig:
    raw = load_json_object(path)
    config = RagBenchConfig.from_dict(raw, runtime=runtime)
    validate_rag_stage(config, stage, has_run_id=bool(raw.get("run_id")))
    return config


def validate_rag_stage(config: RagBenchConfig, stage: str, *, has_run_id: bool) -> None:
    if stage not in STAGES:
        raise ConfigError(f"Unsupported RAG stage: {stage!r}.")
    if stage == "judge" and (config.judge_provider != "vllm" or config.skip_judge):
        raise ConfigError("The judge stage requires judge.provider vllm and skip_judge false.")
    if stage == "judge" and not has_run_id:
        raise ConfigError("The judge stage requires run_id in the JSON configuration.")
    if config.rejudge and stage != "judge":
        raise ConfigError("rag.rejudge requires the judge stage.")
