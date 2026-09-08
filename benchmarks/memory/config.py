"""LoCoMo memory benchmark configuration, loaded from ``configs/memory/accuracy-latency/*.json``."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from inferscale.v1.config import EngineConfig, InferScaleConfig
from inferscale.v1.index.jasper import MAX_JASPER_BEAM_WIDTH

from benchmarks.common.config import (
    DEFAULT_EXTRACTION_LLM_BASE_URL,
    REDACTED,
    ConfigError,
    JudgeConfig,
    RuntimeConfig,
    embedding_api_key,
    expand_path,
    extraction_llm_api_key,
    inferscale_section,
    int_list,
    judge_config,
    optional,
    reject_unknown_keys,
    require,
    section_of,
    stamp_now,
)

from .protocol import (
    ANSWER_PROMPT_PROTOCOL,
    JUDGE_PROMPT_PROTOCOL,
    MEM0AI_VERSION,
    MEMORY_BENCHMARKS_COMMIT,
    MEMORY_BENCHMARKS_REPOSITORY,
    MEMORY_EXTRACTION_MAX_FACTS,
    MEMORY_EXTRACTION_MAX_MODEL_LEN,
    MEMORY_EXTRACTION_MAX_TEXT_CHARS,
    MEMORY_EXTRACTION_MAX_TOKENS,
    MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
    MEMORY_INGESTION_PROTOCOL,
)

AnswerBackend = Literal["kv-injection", "prompt-injection"]
VectorBackend = Literal["jasper", "qdrant"]
ANSWER_BACKENDS = ("kv-injection", "prompt-injection")
VECTOR_BACKENDS = ("jasper", "qdrant")
MEMORY_UNIT = "mem0-fact"
CONTEXT_WINDOW_UNIT = "turns"
CONTEXT_WINDOW_SEMANTICS = "encoding-prefix-discard-v1"
BENCHMARK_NAME = "memory-accuracy-latency"

_TOP_LEVEL_KEYS = (
    "benchmark",
    "model",
    "dataset_path",
    "results_dir",
    "run_id",
    "answer_backend",
    "vector_backend",
    "top_k",
    "context_window",
    "max_samples",
    "max_questions",
    "log_every",
    "log_level",
    "preembed_workers",
    "embedding_cache_enabled",
    "kv_chunk_cache_enabled",
    "skip_judge",
    "rejudge",
    "inferscale",
    "mem0",
    "extraction",
    "judge",
    "sweeps",
)


@dataclass(slots=True, frozen=True)
class ExtractionConfig:
    """How the extract-facts stage serves the answer model for Mem0 extraction."""

    port: int = 8000
    workers: int = 4
    gpu_memory_utilization: float = 0.85
    health_timeout_s: int = 900
    structured_outputs: dict[str, Any] | None = field(
        default_factory=lambda: {"backend": "xgrammar", "disable_any_whitespace": True}
    )
    extra_vllm_args: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"


@dataclass(slots=True, frozen=True)
class SweepVariant:
    name: str
    answer_backend: str
    vector_backend: str
    kv_store_backend: str = "gpu"
    kv_staging_slots: int | None = None
    context_windows: tuple[int, ...] = (0,)


@dataclass(slots=True, frozen=True)
class MemorySweep:
    name: str
    log_dir_prefix: str
    top_k: tuple[int, ...]
    variants: tuple[SweepVariant, ...]


@dataclass(slots=True, frozen=True)
class MemoryCell:
    """One (variant, top_k, context_window) point of a sweep."""

    variant: str
    answer_backend: str
    vector_backend: str
    top_k: int
    context_window: int
    kv_store_backend: str = "gpu"
    kv_staging_slots: int | None = None

    def run_id(self, label: str, max_samples: int | None, stamp: str) -> str:
        samples = str(max_samples) if max_samples is not None else "all"
        return (
            f"{label}-{self.variant}-mem0-{self.vector_backend}{samples}"
            f"-k{self.top_k}-s{self.context_window}-{stamp}"
        )

    def materialize(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Produce one run's JSON from the grid declared in the same JSON file."""
        result = deepcopy(dict(data))
        result.update(
            answer_backend=self.answer_backend,
            vector_backend=self.vector_backend,
            top_k=self.top_k,
            context_window=self.context_window,
        )
        kv = result.setdefault("inferscale", {}).setdefault("kv", {})
        kv["store_backend"] = self.kv_store_backend
        if self.kv_staging_slots is not None:
            kv["staging_slots"] = self.kv_staging_slots
        return result


@dataclass(slots=True)
class MemoryRunConfig:
    inferscale: InferScaleConfig
    label: str
    dataset_path: Path
    results_dir: Path
    run_id: str
    cache_root: Path
    local_store_dir: Path
    embedding_cache_dir: Path
    memory_llm_cache_dir: Path
    judge: JudgeConfig
    answer_backend: str = "kv-injection"
    vector_backend: str = "jasper"
    context_window: int = 0
    max_samples: int | None = None
    max_questions: int | None = None
    log_every: int = 5
    log_level: str = "INFO"
    preembed_workers: int = 4
    embedding_cache_enabled: bool = True
    kv_chunk_cache_enabled: bool = True
    skip_judge: bool = False
    rejudge: bool = False
    memory_llm_provider: str = "vllm"
    memory_llm_base_url: str | None = DEFAULT_EXTRACTION_LLM_BASE_URL
    memory_llm_api_key: str | None = None
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    sweeps: dict[str, MemorySweep] = field(default_factory=dict)
    memory_unit: str = field(default=MEMORY_UNIT, init=False)
    mem0_infer: bool = field(default=True, init=False)
    memory_benchmarks_repository: str = field(default=MEMORY_BENCHMARKS_REPOSITORY, init=False)
    memory_benchmarks_commit: str = field(default=MEMORY_BENCHMARKS_COMMIT, init=False)
    mem0ai_version: str = field(default=MEM0AI_VERSION, init=False)
    memory_extraction_response_protocol: str = field(default=MEMORY_EXTRACTION_RESPONSE_PROTOCOL, init=False)
    memory_extraction_max_model_len: int = field(default=MEMORY_EXTRACTION_MAX_MODEL_LEN, init=False)
    memory_extraction_max_tokens: int = field(default=MEMORY_EXTRACTION_MAX_TOKENS, init=False)
    memory_extraction_max_facts: int = field(default=MEMORY_EXTRACTION_MAX_FACTS, init=False)
    memory_extraction_max_text_chars: int = field(default=MEMORY_EXTRACTION_MAX_TEXT_CHARS, init=False)
    memory_ingestion_protocol: str = field(default=MEMORY_INGESTION_PROTOCOL, init=False)
    answer_prompt_protocol: str = field(default=ANSWER_PROMPT_PROTOCOL, init=False)
    judge_prompt_protocol: str = field(default=JUDGE_PROMPT_PROTOCOL, init=False)
    context_window_unit: str = field(default=CONTEXT_WINDOW_UNIT, init=False)
    context_window_semantics: str = field(default=CONTEXT_WINDOW_SEMANTICS, init=False)

    # ---- library views -------------------------------------------------
    @property
    def model(self) -> str:
        return self.inferscale.model

    @property
    def memory_llm_model(self) -> str:
        """Mem0 fact extraction always uses the answer model."""
        return self.inferscale.model

    @property
    def top_k(self) -> int:
        return self.inferscale.top_k

    @property
    def kv_dtype(self) -> str:
        return self.inferscale.kv.dtype

    @property
    def kv_device(self) -> str:
        return self.inferscale.kv.device

    @property
    def kv_max_position(self) -> int:
        return self.inferscale.kv.max_position

    @property
    def kv_store_backend(self) -> str:
        return self.inferscale.kv.store_backend

    @property
    def kv_staging_slots(self) -> int:
        return self.inferscale.kv.staging_slots

    @property
    def kv_block_size(self) -> int:
        return self.inferscale.engine.block_size

    @property
    def kv_max_model_len(self) -> int:
        return self.inferscale.engine.max_model_len

    @property
    def kv_gpu_memory_utilization(self) -> float:
        return self.inferscale.engine.gpu_memory_utilization

    @property
    def kv_enable_prefix_caching(self) -> bool:
        return self.inferscale.engine.enable_prefix_caching

    @property
    def kv_connector_module(self) -> str:
        return self.inferscale.engine.connector_module

    @property
    def jasper_n_neighbors(self) -> int:
        return self.inferscale.index.n_neighbors

    @property
    def jasper_alpha(self) -> float:
        return self.inferscale.index.alpha

    @property
    def jasper_workspace_budget(self) -> str:
        return self.inferscale.index.workspace_budget

    @property
    def jasper_beam_width(self) -> int:
        return self.inferscale.index.beam_width

    @property
    def jasper_effective_beam_width(self) -> int | None:
        if self.vector_backend != "jasper":
            return None
        return max(self.jasper_beam_width, self.top_k)

    @property
    def embedding_model(self) -> str:
        return self.inferscale.embedding.model

    @property
    def embedding_base_url(self) -> str | None:
        return self.inferscale.embedding.base_url

    @property
    def embedding_api_key(self) -> str | None:
        return embedding_api_key()

    @property
    def temperature(self) -> float:
        return self.inferscale.generation.temperature

    @property
    def top_p(self) -> float:
        return self.inferscale.generation.top_p

    @property
    def max_answer_tokens(self) -> int:
        return self.inferscale.generation.max_tokens

    @property
    def judge_provider(self) -> str:
        return self.judge.provider

    @property
    def judge_model(self) -> str:
        return self.judge.model

    @property
    def judge_base_url(self) -> str | None:
        return self.judge.base_url

    @property
    def judge_api_key(self) -> str | None:
        return self.judge.api_key

    @property
    def max_judge_tokens(self) -> int:
        return self.judge.max_tokens

    @property
    def with_evidence(self) -> bool:
        return self.judge.with_evidence

    @with_evidence.setter
    def with_evidence(self, value: bool) -> None:
        self.judge.with_evidence = bool(value)

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id

    def engine_config(self) -> EngineConfig:
        return self.inferscale.engine

    def local_store_scratch_dir(self, run_id: str | None = None) -> Path:
        return self.local_store_dir / "inferscale-bench-stores" / (run_id or self.run_id)

    def to_jsonable(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "benchmark": BENCHMARK_NAME,
            "model": self.model,
            "label": self.label,
            "dataset_path": str(self.dataset_path),
            "results_dir": str(self.results_dir),
            "run_id": self.run_id,
            "answer_backend": self.answer_backend,
            "vector_backend": self.vector_backend,
            "top_k": self.top_k,
            "context_window": self.context_window,
            "max_samples": self.max_samples,
            "max_questions": self.max_questions,
            "log_every": self.log_every,
            "log_level": self.log_level,
            "preembed_workers": self.preembed_workers,
            "embedding_cache_enabled": self.embedding_cache_enabled,
            "embedding_cache_dir": str(self.embedding_cache_dir),
            "kv_chunk_cache_enabled": self.kv_chunk_cache_enabled,
            "cache_root": str(self.cache_root),
            "local_store_dir": str(self.local_store_dir),
            "skip_judge": self.skip_judge,
            "rejudge": self.rejudge,
            "memory_llm_provider": self.memory_llm_provider,
            "memory_llm_model": self.memory_llm_model,
            "memory_llm_base_url": self.memory_llm_base_url,
            "memory_llm_api_key": REDACTED if self.memory_llm_api_key else None,
            "memory_llm_cache_dir": str(self.memory_llm_cache_dir),
            "embedding_api_key": REDACTED if self.embedding_api_key else None,
            "inferscale": self.inferscale.to_dict(),
            "judge": self.judge.to_jsonable(),
            "with_evidence": self.with_evidence,
            "jasper_effective_beam_width": self.jasper_effective_beam_width,
            "extraction": _extraction_dict(self.extraction),
            "sweeps": {name: _sweep_dict(sweep) for name, sweep in self.sweeps.items()},
            "memory_unit": self.memory_unit,
            "mem0_infer": self.mem0_infer,
            "memory_benchmarks_repository": self.memory_benchmarks_repository,
            "memory_benchmarks_commit": self.memory_benchmarks_commit,
            "mem0ai_version": self.mem0ai_version,
            "memory_extraction_response_protocol": self.memory_extraction_response_protocol,
            "memory_extraction_max_model_len": self.memory_extraction_max_model_len,
            "memory_extraction_max_tokens": self.memory_extraction_max_tokens,
            "memory_extraction_max_facts": self.memory_extraction_max_facts,
            "memory_extraction_max_text_chars": self.memory_extraction_max_text_chars,
            "memory_ingestion_protocol": self.memory_ingestion_protocol,
            "answer_prompt_protocol": self.answer_prompt_protocol,
            "judge_prompt_protocol": self.judge_prompt_protocol,
            "context_window_unit": self.context_window_unit,
            "context_window_semantics": self.context_window_semantics,
        }
        return data

    # ---- loading -------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        runtime: RuntimeConfig,
        where: str = "memory config",
    ) -> "MemoryRunConfig":
        merged = dict(data)
        reject_unknown_keys(merged, _TOP_LEVEL_KEYS, where)
        benchmark = optional(merged, "benchmark", str, BENCHMARK_NAME, where)
        if benchmark != BENCHMARK_NAME:
            raise ConfigError(f"{where}.benchmark must be {BENCHMARK_NAME!r}, got {benchmark!r}.")

        alias = require(merged, "model", str, where)
        model = runtime.resolve_model(alias)
        label = runtime.model_label(alias)
        answer_backend = optional(merged, "answer_backend", str, "kv-injection", where)
        if answer_backend not in ANSWER_BACKENDS:
            raise ConfigError(f"{where}.answer_backend must be one of {ANSWER_BACKENDS}.")
        vector_backend = optional(merged, "vector_backend", str, "jasper", where)
        if vector_backend not in VECTOR_BACKENDS:
            raise ConfigError(f"{where}.vector_backend must be one of {VECTOR_BACKENDS}.")
        top_k = optional(merged, "top_k", int, 50, where)
        if top_k < 1:
            raise ConfigError(f"{where}.top_k must be >= 1.")
        context_window = optional(merged, "context_window", int, 0, where)
        if context_window < 0:
            raise ConfigError(f"{where}.context_window must be >= 0.")
        preembed_workers = optional(merged, "preembed_workers", int, 4, where)
        if preembed_workers < 1:
            raise ConfigError(f"{where}.preembed_workers must be >= 1.")
        log_every = optional(merged, "log_every", int, 5, where)
        if log_every < 0:
            raise ConfigError(f"{where}.log_every must be >= 0 (0 disables progress logging).")
        max_samples = optional(merged, "max_samples", int, None, where)
        max_questions = optional(merged, "max_questions", int, None, where)
        for name, value in (("max_samples", max_samples), ("max_questions", max_questions)):
            if value is not None and value < 1:
                raise ConfigError(f"{where}.{name} must be >= 1.")

        inferscale = inferscale_section(merged, model=model, top_k=top_k, where=where)
        if vector_backend == "jasper" and max(inferscale.index.beam_width, top_k) > MAX_JASPER_BEAM_WIDTH:
            raise ConfigError(
                f"{where}: effective Jasper beam width must be <= {MAX_JASPER_BEAM_WIDTH}; "
                f"got max({inferscale.index.beam_width}, {top_k})."
            )
        if answer_backend == "prompt-injection" and not inferscale.engine.enable_prefix_caching:
            raise ConfigError(f"{where}: answer_backend prompt-injection requires inferscale.engine.enable_prefix_caching.")

        judge = judge_config(
            section_of(merged, "judge", where, required=False),
            runtime,
            where=f"{where}.judge",
        )
        skip_judge = optional(merged, "skip_judge", bool, False, where) or judge.provider == "none"
        rejudge = optional(merged, "rejudge", bool, False, where)

        mem0_section = section_of(merged, "mem0", where, required=False)
        reject_unknown_keys(mem0_section, ("llm_base_url",), f"{where}.mem0")
        memory_llm_base_url = optional(
            mem0_section, "llm_base_url", str, DEFAULT_EXTRACTION_LLM_BASE_URL, f"{where}.mem0"
        )

        extraction = _extraction_config(section_of(merged, "extraction", where, required=False), f"{where}.extraction")
        endpoint_port = urlsplit(memory_llm_base_url).port
        if endpoint_port is not None and endpoint_port != extraction.port:
            raise ConfigError(
                f"{where}: mem0.llm_base_url port {endpoint_port} must equal extraction.port {extraction.port}; "
                "the fact-catalog identity records the extraction endpoint."
            )

        sweeps = _sweeps(section_of(merged, "sweeps", where, required=False), f"{where}.sweeps")

        layout = runtime.layout
        dataset_path = expand_path(optional(merged, "dataset_path", str, "data/locomo10.json", where), root=runtime.root)
        results_value = optional(merged, "results_dir", str, None, where)
        results_dir = expand_path(results_value, root=runtime.root) if results_value else layout.results_root
        run_id = optional(merged, "run_id", str, None, where)
        if run_id is not None and (
            not run_id.strip() or run_id in {".", ".."} or "/" in run_id or "\\" in run_id
        ):
            raise ConfigError(f"{where}.run_id must be a non-empty directory name.")
        run_id = run_id or stamp_now()

        return cls(
            inferscale=inferscale,
            label=label,
            dataset_path=dataset_path,
            results_dir=results_dir,
            run_id=run_id,
            cache_root=layout.cache_root,
            local_store_dir=layout.local_store_dir,
            embedding_cache_dir=layout.embedding_cache_dir,
            memory_llm_cache_dir=layout.memory_llm_cache_dir,
            judge=judge,
            answer_backend=answer_backend,
            vector_backend=vector_backend,
            context_window=context_window,
            max_samples=max_samples,
            max_questions=max_questions,
            log_every=log_every,
            log_level=optional(merged, "log_level", str, "INFO", where),
            preembed_workers=preembed_workers,
            embedding_cache_enabled=optional(merged, "embedding_cache_enabled", bool, True, where),
            kv_chunk_cache_enabled=optional(merged, "kv_chunk_cache_enabled", bool, True, where),
            skip_judge=skip_judge,
            rejudge=rejudge,
            memory_llm_base_url=memory_llm_base_url,
            memory_llm_api_key=extraction_llm_api_key(),
            extraction=extraction,
            sweeps=sweeps,
        )

    def sweep(self, name: str) -> MemorySweep:
        try:
            return self.sweeps[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.sweeps)) or "none"
            raise ConfigError(f"Unknown sweep {name!r}; configured sweeps: {known}.") from exc

    def sweep_cells(self, name: str) -> list[MemoryCell]:
        """Expand a sweep in the order top_k, variant, context_window."""
        sweep = self.sweep(name)
        cells: list[MemoryCell] = []
        for top_k in sweep.top_k:
            for variant in sweep.variants:
                for window in variant.context_windows:
                    cells.append(
                        MemoryCell(
                            variant=variant.name,
                            answer_backend=variant.answer_backend,
                            vector_backend=variant.vector_backend,
                            top_k=top_k,
                            context_window=window,
                            kv_store_backend=variant.kv_store_backend,
                            kv_staging_slots=variant.kv_staging_slots,
                        )
                    )
        return cells

    def context_windows(self) -> tuple[int, ...]:
        """Every context window any configured sweep encodes, for the precompute-kv stage."""
        windows: list[int] = [self.context_window]
        for sweep in self.sweeps.values():
            for variant in sweep.variants:
                if variant.answer_backend == "kv-injection":
                    windows.extend(variant.context_windows)
        return tuple(sorted(set(windows)))


def load_memory_config(
    path: str | Path,
    runtime: RuntimeConfig,
    *,
    stage: str = "run",
) -> MemoryRunConfig:
    from benchmarks.common.config import load_json_object

    data = load_json_object(path)
    config = MemoryRunConfig.from_dict(data, runtime=runtime, where=f"memory config {path}")
    validate_memory_stage(config, stage, has_run_id=bool(data.get("run_id")))
    return config


def validate_memory_stage(config: MemoryRunConfig, stage: str, *, has_run_id: bool) -> None:
    if stage not in {"run", "judge", "check-catalogs", "preembed", "precompute-kv"}:
        raise ConfigError(f"Unsupported memory stage: {stage!r}.")
    if stage == "judge" and not has_run_id:
        raise ConfigError("The judge stage requires run_id in JSON to select an existing run.")
    if config.rejudge and stage != "judge":
        raise ConfigError("JSON rejudge=true requires the judge stage.")
    if stage == "judge" and config.skip_judge:
        raise ConfigError("The judge stage requires judging to be enabled in JSON.")


def _extraction_config(section: Mapping[str, Any], where: str) -> ExtractionConfig:
    reject_unknown_keys(
        section,
        ("port", "workers", "gpu_memory_utilization", "health_timeout_s", "structured_outputs", "extra_vllm_args"),
        where,
    )
    defaults = ExtractionConfig()
    port = optional(section, "port", int, defaults.port, where)
    workers = optional(section, "workers", int, defaults.workers, where)
    gpu = optional(section, "gpu_memory_utilization", (int, float), defaults.gpu_memory_utilization, where)
    timeout = optional(section, "health_timeout_s", int, defaults.health_timeout_s, where)
    if port < 1 or workers < 1 or timeout < 1 or not 0 < gpu <= 1:
        raise ConfigError(f"{where}: port, workers, and health_timeout_s must be >= 1 and gpu_memory_utilization in (0, 1].")
    structured = section.get("structured_outputs", defaults.structured_outputs)
    if structured is not None and not isinstance(structured, dict):
        raise ConfigError(f"{where}.structured_outputs must be an object or null.")
    extra = section.get("extra_vllm_args", [])
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ConfigError(f"{where}.extra_vllm_args must be a list of strings.")
    return ExtractionConfig(
        port=port,
        workers=workers,
        gpu_memory_utilization=float(gpu),
        health_timeout_s=timeout,
        structured_outputs=dict(structured) if structured is not None else None,
        extra_vllm_args=tuple(extra),
    )


def _extraction_dict(extraction: ExtractionConfig) -> dict[str, Any]:
    return {
        "port": extraction.port,
        "workers": extraction.workers,
        "gpu_memory_utilization": extraction.gpu_memory_utilization,
        "health_timeout_s": extraction.health_timeout_s,
        "structured_outputs": extraction.structured_outputs,
        "extra_vllm_args": list(extraction.extra_vllm_args),
    }


def _sweeps(section: Mapping[str, Any], where: str) -> dict[str, MemorySweep]:
    sweeps: dict[str, MemorySweep] = {}
    for name, raw in section.items():
        sweep_where = f"{where}.{name}"
        if not isinstance(raw, dict):
            raise ConfigError(f"{sweep_where} must be an object.")
        reject_unknown_keys(raw, ("log_dir_prefix", "top_k", "variants"), sweep_where)
        variants_raw = raw.get("variants")
        if not isinstance(variants_raw, list) or not variants_raw:
            raise ConfigError(f"{sweep_where}.variants must be a non-empty list.")
        variants: list[SweepVariant] = []
        seen: set[str] = set()
        for index, item in enumerate(variants_raw):
            item_where = f"{sweep_where}.variants[{index}]"
            if not isinstance(item, dict):
                raise ConfigError(f"{item_where} must be an object.")
            reject_unknown_keys(
                item,
                ("name", "answer_backend", "vector_backend", "kv_store_backend", "kv_staging_slots", "context_window"),
                item_where,
            )
            variant_name = require(item, "name", str, item_where)
            if variant_name in seen:
                raise ConfigError(f"{sweep_where} has a duplicate variant name {variant_name!r}.")
            seen.add(variant_name)
            answer_backend = require(item, "answer_backend", str, item_where)
            vector_backend = require(item, "vector_backend", str, item_where)
            if answer_backend not in ANSWER_BACKENDS or vector_backend not in VECTOR_BACKENDS:
                raise ConfigError(f"{item_where}: unknown answer or vector backend.")
            kv_store_backend = optional(item, "kv_store_backend", str, "gpu", item_where)
            if kv_store_backend not in ("gpu", "cpu"):
                raise ConfigError(f"{item_where}.kv_store_backend must be gpu or cpu.")
            staging = optional(item, "kv_staging_slots", int, None, item_where)
            if staging is not None and staging < 1:
                raise ConfigError(f"{item_where}.kv_staging_slots must be >= 1.")
            windows = int_list(item, "context_window", item_where, default=(0,))
            if any(window < 0 for window in windows):
                raise ConfigError(f"{item_where}.context_window entries must be >= 0.")
            variants.append(
                SweepVariant(
                    name=variant_name,
                    answer_backend=answer_backend,
                    vector_backend=vector_backend,
                    kv_store_backend=kv_store_backend,
                    kv_staging_slots=staging,
                    context_windows=windows,
                )
            )
        top_ks = int_list(raw, "top_k", sweep_where)
        if any(top_k < 1 for top_k in top_ks):
            raise ConfigError(f"{sweep_where}.top_k entries must be >= 1.")
        sweeps[str(name)] = MemorySweep(
            name=str(name),
            log_dir_prefix=optional(raw, "log_dir_prefix", str, f"{name}-logs", sweep_where),
            top_k=top_ks,
            variants=tuple(variants),
        )
    return sweeps


def _sweep_dict(sweep: MemorySweep) -> dict[str, Any]:
    return {
        "log_dir_prefix": sweep.log_dir_prefix,
        "top_k": list(sweep.top_k),
        "variants": [
            {
                "name": variant.name,
                "answer_backend": variant.answer_backend,
                "vector_backend": variant.vector_backend,
                "kv_store_backend": variant.kv_store_backend,
                "kv_staging_slots": variant.kv_staging_slots,
                "context_window": list(variant.context_windows),
            }
            for variant in sweep.variants
        ],
    }
