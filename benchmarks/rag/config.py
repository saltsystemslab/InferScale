from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchmarks.common.config import (
    ConfigError,
    JudgeConfig,
    RuntimeConfig,
    embedding_api_key,
    expand_path,
    inferscale_section,
    int_list,
    judge_config,
    load_json_object,
    load_runtime_config,
    optional,
    reject_unknown_keys,
    section_of,
    stamp_now,
    str_list,
)
from inferscale.v1.config import EmbeddingConfig, EngineConfig
from inferscale.v1.index.jasper import MAX_JASPER_BEAM_WIDTH

AnswerBackend = Literal["kv-injection", "prompt-injection"]
JudgeProvider = Literal["vllm", "none"]

DEFAULT_DATASET = "multihoprag"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CONTEXT_WINDOW = 5
DEFAULT_TOP_K = 15
DEFAULT_MAX_ANSWER_TOKENS = 64
DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_EMBEDDING_MODEL = EmbeddingConfig().model
DEFAULT_JUDGE_PROVIDER = JudgeConfig().provider
DEFAULT_MAX_JUDGE_TOKENS = JudgeConfig().max_tokens
# Parse-time slack for the scaffold and the templated question on top of the
# retrieved chunks; the answer path still enforces the exact budget per query.
MEMORY_BUDGET_MARGIN_TOKENS = 512


@dataclass(slots=True)
class RagBenchConfig:
    runtime: InitVar[RuntimeConfig | None] = None
    dataset_name: str = DEFAULT_DATASET
    data_dir: Path | None = None
    results_dir: Path = field(default_factory=lambda: load_runtime_config().layout.results_root)
    run_id: str = field(default_factory=stamp_now)

    model: str = "llama"
    answer_backend: AnswerBackend = "kv-injection"

    chunk_size: int = DEFAULT_CHUNK_SIZE
    context_window: int = DEFAULT_CONTEXT_WINDOW
    top_k: int = DEFAULT_TOP_K

    judge_provider: JudgeProvider = DEFAULT_JUDGE_PROVIDER
    judge_model: str = field(default_factory=lambda: load_runtime_config().judge_server.model)
    judge_base_url: str | None = field(
        default_factory=lambda: load_runtime_config().judge_server.base_url
    )
    judge_api_key: str | None = field(default_factory=lambda: load_runtime_config().judge_api_key())

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str | None = None
    embedding_api_key: str | None = field(default_factory=embedding_api_key)
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = field(default_factory=lambda: load_runtime_config().layout.embedding_cache_dir)
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE

    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    temperature: float = 0.0
    top_p: float = 1.0
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
    max_judge_tokens: int = DEFAULT_MAX_JUDGE_TOKENS

    kv_connector_module: str = "inferscale.v1.kv.connector"
    kv_gpu_memory_utilization: float = 0.40
    kv_block_size: int = 16
    kv_max_model_len: int = 32768
    kv_max_position: int = 32768
    kv_dtype: str = "bfloat16"
    kv_device: str = "cuda:0"
    kv_enable_prefix_caching: bool = True
    kv_chunk_cache_root: Path | None = None
    # The corpus chunk KV is fully host-RAM resident at answer time, loaded
    # once from the precompute disk cache; there is no answer-time disk or
    # GPU-resident corpus backend.
    kv_store_backend: Literal["cpu"] = field(default="cpu", init=False)

    max_queries: int | None = None
    log_every: int = 25
    log_level: str = "INFO"
    skip_judge: bool = False
    rejudge: bool = False

    def __post_init__(self, runtime: RuntimeConfig | None) -> None:
        if self.data_dir is None:
            self.data_dir = Path("data") / self.dataset_name
        self.model = (runtime or load_runtime_config()).resolve_model(self.model)

    def result_mode(self) -> str:
        return "rag-kv" if self.answer_backend == "kv-injection" else "rag-prefix"

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

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        data["mode"] = self.result_mode()
        data["jasper_effective_beam_width"] = self.jasper_effective_beam_width
        for key in ("data_dir", "results_dir", "embedding_cache_dir", "kv_chunk_cache_root"):
            if data[key] is not None:
                data[key] = str(data[key])
        for key in ("judge_api_key", "embedding_api_key"):
            if data.get(key):
                data[key] = "<redacted>"
        return data

    @classmethod
    def from_json_file(
        cls, path: str | Path, *, runtime: RuntimeConfig | None = None
    ) -> "RagBenchConfig":
        return cls.from_dict(load_json_object(path), runtime=runtime)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, runtime: RuntimeConfig | None = None
    ) -> "RagBenchConfig":
        runtime = runtime or load_runtime_config()
        where = "rag"
        reject_unknown_keys(
            data,
            (
                "benchmark", "dataset", "data_dir", "models", "model", "results_dir",
                "run_id", "answer_backend", "chunk_size", "context_window", "top_k",
                "max_queries", "log_every", "log_level", "embed_batch_size",
                "embedding_cache_enabled", "embedding_cache_dir", "kv_chunk_cache_root",
                "skip_judge", "rejudge", "inferscale", "judge", "sweeps",
            ),
            where,
        )
        if optional(data, "benchmark", str, "rag", where) != "rag":
            raise ConfigError("rag.benchmark must be 'rag'.")
        if "model" in data and "models" in data:
            raise ConfigError("rag must specify model or models, not both.")
        models = (
            [optional(data, "model", str, "llama", where)]
            if "models" not in data else str_list(data, "models", where)
        )
        if len(models) != 1:
            raise ConfigError("A RAG invocation requires one model; use the sweep launcher for multiple models.")
        model = runtime.resolve_model(models[0])
        top_k = optional(data, "top_k", int, DEFAULT_TOP_K, where)
        raw_library = dict(section_of(data, "inferscale", where, required=False))
        reject_unknown_keys(raw_library, ("engine", "kv", "index", "embedding", "generation"), "rag.inferscale")
        # RAG has a shorter generation cap and host-resident corpus KV.
        raw_library["generation"] = {
            "max_tokens": DEFAULT_MAX_ANSWER_TOKENS,
            **section_of(raw_library, "generation", "rag.inferscale", required=False),
        }
        raw_library["kv"] = {
            "store_backend": "cpu",
            **section_of(raw_library, "kv", "rag.inferscale", required=False),
        }
        library = inferscale_section({"inferscale": raw_library}, model=model, top_k=top_k, where=where)
        if library.kv.store_backend != "cpu":
            raise ConfigError("rag.inferscale.kv.store_backend must be cpu.")
        judge = judge_config(section_of(data, "judge", where, required=False), runtime, where="rag.judge")
        if judge.with_evidence:
            raise ConfigError("rag.judge.with_evidence is not supported by the RAG judge.")
        skip_judge = optional(data, "skip_judge", bool, False, where)
        dataset_name = optional(data, "dataset", str, DEFAULT_DATASET, where)
        from benchmarks.rag.datasets import get_dataset

        try:
            dataset_name = get_dataset(dataset_name).name
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

        def path_value(key: str, default: Path | None) -> Path | None:
            value = optional(data, key, str, None, where)
            return expand_path(value, root=runtime.root) if value is not None else default

        config = cls(
            runtime=runtime,
            dataset_name=dataset_name,
            data_dir=path_value("data_dir", runtime.root / "data" / dataset_name),
            results_dir=path_value("results_dir", runtime.layout.results_root),
            run_id=optional(data, "run_id", str, stamp_now(), where),
            model=models[0],
            answer_backend=optional(data, "answer_backend", str, "kv-injection", where),
            chunk_size=optional(data, "chunk_size", int, DEFAULT_CHUNK_SIZE, where),
            context_window=optional(data, "context_window", int, DEFAULT_CONTEXT_WINDOW, where),
            top_k=top_k,
            judge_provider="none" if skip_judge else judge.provider,
            judge_model=judge.model,
            judge_base_url=judge.base_url,
            judge_api_key=judge.api_key,
            embedding_model=library.embedding.model,
            embedding_base_url=library.embedding.base_url,
            embedding_api_key=embedding_api_key(),
            embedding_cache_enabled=optional(data, "embedding_cache_enabled", bool, True, where),
            embedding_cache_dir=path_value("embedding_cache_dir", runtime.layout.embedding_cache_dir),
            embed_batch_size=optional(data, "embed_batch_size", int, library.embedding.batch_size, where),
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
            kv_chunk_cache_root=path_value("kv_chunk_cache_root", runtime.layout.rag_kv_chunk_cache_root),
            max_queries=optional(data, "max_queries", int, None, where),
            log_every=optional(data, "log_every", int, 25, where),
            log_level=optional(data, "log_level", str, "INFO", where).upper(),
            skip_judge=skip_judge or judge.provider == "none",
            rejudge=optional(data, "rejudge", bool, False, where),
        )
        _validate(config)
        _validate_sweeps(section_of(data, "sweeps", where, required=False))
        return config


def _validate(config: RagBenchConfig) -> None:
    if config.answer_backend not in {"kv-injection", "prompt-injection"}:
        raise ConfigError("rag.answer_backend must be kv-injection or prompt-injection.")
    for name in ("top_k", "chunk_size", "embed_batch_size", "log_every", "max_answer_tokens"):
        if getattr(config, name) < 1:
            raise ConfigError(f"rag.{name} must be >= 1.")
    if config.context_window < 0:
        raise ConfigError("rag.context_window must be >= 0.")
    if config.max_queries is not None and config.max_queries < 1:
        raise ConfigError("rag.max_queries must be >= 1 or null.")
    if not config.run_id.strip() or Path(config.run_id).name != config.run_id or config.run_id in {".", ".."}:
        raise ConfigError("rag.run_id must be a non-empty directory name.")
    if config.log_level not in {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError("rag.log_level is not a supported logging level.")
    if config.answer_backend == "prompt-injection" and not config.kv_enable_prefix_caching:
        raise ConfigError("rag.answer_backend prompt-injection requires inferscale.engine.enable_prefix_caching.")
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
        reject_unknown_keys(sweep, ("log_dir_prefix", "top_k", "answer_backends"), where)
        optional(sweep, "log_dir_prefix", str, "rag-sweep-logs", where)
        if any(k < 1 or k > MAX_JASPER_BEAM_WIDTH for k in int_list(sweep, "top_k", where, default=[DEFAULT_TOP_K])):
            raise ConfigError(f"{where}.top_k must be between 1 and {MAX_JASPER_BEAM_WIDTH}.")
        backends = str_list(sweep, "answer_backends", where, default=["kv-injection", "prompt-injection"])
        if any(backend not in {"kv-injection", "prompt-injection"} for backend in backends):
            raise ConfigError(f"{where}.answer_backends must contain only kv-injection or prompt-injection.")


def load_rag_config(
    path: str | Path, *, runtime: RuntimeConfig | None = None, stage: str = "run"
) -> RagBenchConfig:
    raw = load_json_object(path)
    config = RagBenchConfig.from_dict(raw, runtime=runtime)
    validate_rag_stage(config, stage, has_run_id=bool(raw.get("run_id")))
    return config


def validate_rag_stage(config: RagBenchConfig, stage: str, *, has_run_id: bool) -> None:
    if stage not in {"estimate", "preembed", "precompute-kv", "run", "judge"}:
        raise ConfigError(f"Unsupported RAG stage: {stage!r}.")
    if stage == "judge" and (config.judge_provider != "vllm" or config.skip_judge):
        raise ConfigError("The judge stage requires judge.provider vllm and skip_judge false.")
    if stage == "judge" and not has_run_id:
        raise ConfigError("The judge stage requires run_id in the JSON configuration.")
    if config.rejudge and stage != "judge":
        raise ConfigError("rag.rejudge requires the judge stage.")
