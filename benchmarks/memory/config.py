from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from benchmarks.common.config import (
    DEFAULT_EXTRACTION_LLM_BASE_URL,
    REDACTED,
    ConfigError,
    RuntimeConfig,
    extraction_llm_api_key,
    load_json_object,
    stamp_now,
)
from benchmarks.common.inferscale_config import inferscale_section
from benchmarks.common.json_fields import (
    expand_path,
    optional,
    reject_unknown_keys,
    require,
    section_of,
)
from benchmarks.common.judge_config import JudgeConfig, judge_config
from benchmarks.common.lmcache.config import LMCacheConfig, parse_lmcache_config
from inferscale.v1.config import EngineConfig, InferScaleConfig
from inferscale.v1.index.jasper import MAX_JASPER_BEAM_WIDTH

from .extraction_config import ExtractionConfig, extraction_dict, parse_extraction_config
from .protocol import (
    ANSWER_BACKENDS,
    LMCACHE_BACKEND,
    PROMPT_INJECTION_BACKEND,
    PROTOCOL_STAMP,
    VECTOR_BACKENDS,
)
from .sweeps import (
    MemoryCell,
    MemorySweep,
    kv_context_windows,
    parse_sweeps,
    sweep_cells,
    sweep_dict,
    sweeps_use_backend,
)

BENCHMARK_NAME = "memory-accuracy-latency"
MEMORY_STAGES = ("run", "judge", "check-catalogs", "preembed", "precompute-kv")
DEFAULT_DATASET_PATH = "data/locomo10.json"

RUN_KEYS = (
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
    "lmcache",
    "sweeps",
)
MEM0_KEYS = ("llm_base_url",)


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
    lmcache: LMCacheConfig | None = None

    @property
    def model(self) -> str:
        return self.inferscale.model

    @property
    def memory_llm_model(self) -> str:
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

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id

    def engine_config(self) -> EngineConfig:
        return self.inferscale.engine

    def lmcache_settings(self) -> LMCacheConfig:
        if self.lmcache is None:
            raise RuntimeError(
                f"This run has no lmcache section; it is not an {LMCACHE_BACKEND} run."
            )
        return self.lmcache

    def local_store_scratch_dir(self, run_id: str | None = None) -> Path:
        return self.local_store_dir / "inferscale-bench-stores" / (run_id or self.run_id)

    def sweep_cells(self, name: str) -> list[MemoryCell]:
        if name not in self.sweeps:
            configured_names = ", ".join(sorted(self.sweeps)) or "none"
            raise ConfigError(f"Unknown sweep {name!r}; configured sweeps: {configured_names}.")
        return sweep_cells(self.sweeps[name])

    def context_windows(self) -> tuple[int, ...]:
        return kv_context_windows(self.sweeps.values(), self.context_window)

    def to_jsonable(self) -> dict[str, Any]:
        return {
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
            "inferscale": self.inferscale.to_dict(),
            "judge": self.judge.to_jsonable(),
            "with_evidence": self.with_evidence,
            "jasper_effective_beam_width": self.jasper_effective_beam_width,
            "extraction": extraction_dict(self.extraction),
            "lmcache": None if self.lmcache is None else self.lmcache.to_jsonable(),
            "sweeps": {name: sweep_dict(sweep) for name, sweep in self.sweeps.items()},
            **PROTOCOL_STAMP,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        runtime: RuntimeConfig,
        where: str = "memory config",
    ) -> MemoryRunConfig:
        settings = dict(data)
        reject_unknown_keys(settings, RUN_KEYS, where)
        _check_benchmark_name(settings, where)
        authored_model = require(settings, "model", str, where)
        answer_backend, vector_backend = _backends(settings, where)
        limits = _RunLimits.parse(settings, where)
        inferscale = inferscale_section(
            settings,
            model=runtime.resolve_model(authored_model),
            top_k=limits.top_k,
            context_window=limits.context_window,
            where=where,
        )
        _check_backend_requirements(
            inferscale, answer_backend, vector_backend, limits.top_k, limits.context_window, where
        )
        sweeps = parse_sweeps(
            section_of(settings, "sweeps", where, required=False), f"{where}.sweeps"
        )
        judge = judge_config(
            section_of(settings, "judge", where, required=False), runtime, where=f"{where}.judge"
        )
        extraction = parse_extraction_config(
            section_of(settings, "extraction", where, required=False), f"{where}.extraction"
        )
        layout = runtime.layout
        return cls(
            inferscale=inferscale,
            label=runtime.model_label(authored_model),
            dataset_path=_dataset_path(settings, runtime, where),
            results_dir=_results_dir(settings, runtime, where),
            run_id=_run_id(settings, where),
            cache_root=layout.cache_root,
            local_store_dir=layout.local_store_dir,
            embedding_cache_dir=layout.embedding_cache_dir,
            memory_llm_cache_dir=layout.memory_llm_cache_dir,
            judge=judge,
            answer_backend=answer_backend,
            vector_backend=vector_backend,
            context_window=limits.context_window,
            max_samples=limits.max_samples,
            max_questions=limits.max_questions,
            log_every=limits.log_every,
            log_level=optional(settings, "log_level", str, "INFO", where),
            preembed_workers=limits.preembed_workers,
            embedding_cache_enabled=optional(
                settings, "embedding_cache_enabled", bool, True, where
            ),
            kv_chunk_cache_enabled=optional(settings, "kv_chunk_cache_enabled", bool, True, where),
            skip_judge=optional(settings, "skip_judge", bool, False, where)
            or judge.provider == "none",
            rejudge=optional(settings, "rejudge", bool, False, where),
            memory_llm_base_url=_extraction_endpoint(settings, extraction, where),
            memory_llm_api_key=extraction_llm_api_key(),
            extraction=extraction,
            sweeps=sweeps,
            lmcache=_lmcache_settings(settings, answer_backend, sweeps, where),
        )


def load_memory_config(
    path: str | Path, runtime: RuntimeConfig, *, stage: str = "run"
) -> MemoryRunConfig:
    settings = load_json_object(path)
    config = MemoryRunConfig.from_dict(settings, runtime=runtime, where=f"memory config {path}")
    validate_memory_stage(config, stage, has_run_id=bool(settings.get("run_id")))
    return config


def validate_memory_stage(config: MemoryRunConfig, stage: str, *, has_run_id: bool) -> None:
    if stage not in MEMORY_STAGES:
        raise ConfigError(f"Unsupported memory stage: {stage!r}.")
    if stage == "judge" and not has_run_id:
        raise ConfigError("The judge stage requires run_id in JSON to select an existing run.")
    if config.rejudge and stage != "judge":
        raise ConfigError("JSON rejudge=true requires the judge stage.")
    if stage == "judge" and config.skip_judge:
        raise ConfigError("The judge stage requires judging to be enabled in JSON.")


def _check_benchmark_name(settings: Mapping[str, Any], where: str) -> None:
    benchmark = optional(settings, "benchmark", str, BENCHMARK_NAME, where)
    if benchmark != BENCHMARK_NAME:
        raise ConfigError(f"{where}.benchmark must be {BENCHMARK_NAME!r}, got {benchmark!r}.")


@dataclass(slots=True, frozen=True)
class _RunLimits:
    top_k: int
    context_window: int
    preembed_workers: int
    log_every: int
    max_samples: int | None
    max_questions: int | None

    @classmethod
    def parse(cls, settings: Mapping[str, Any], where: str) -> _RunLimits:
        top_k = optional(settings, "top_k", int, 50, where)
        if top_k < 1:
            raise ConfigError(f"{where}.top_k must be >= 1.")
        context_window = optional(settings, "context_window", int, 0, where)
        if context_window < 0:
            raise ConfigError(f"{where}.context_window must be >= 0.")
        preembed_workers = optional(settings, "preembed_workers", int, 4, where)
        if preembed_workers < 1:
            raise ConfigError(f"{where}.preembed_workers must be >= 1.")
        log_every = optional(settings, "log_every", int, 5, where)
        if log_every < 0:
            raise ConfigError(f"{where}.log_every must be >= 0 (0 disables progress logging).")
        max_samples = optional(settings, "max_samples", int, None, where)
        max_questions = optional(settings, "max_questions", int, None, where)
        for key, limit in (("max_samples", max_samples), ("max_questions", max_questions)):
            if limit is not None and limit < 1:
                raise ConfigError(f"{where}.{key} must be >= 1.")
        return cls(top_k, context_window, preembed_workers, log_every, max_samples, max_questions)


def _backends(settings: Mapping[str, Any], where: str) -> tuple[str, str]:
    answer_backend = optional(settings, "answer_backend", str, "kv-injection", where)
    if answer_backend not in ANSWER_BACKENDS:
        raise ConfigError(f"{where}.answer_backend must be one of {ANSWER_BACKENDS}.")
    vector_backend = optional(settings, "vector_backend", str, "jasper", where)
    if vector_backend not in VECTOR_BACKENDS:
        raise ConfigError(f"{where}.vector_backend must be one of {VECTOR_BACKENDS}.")
    return answer_backend, vector_backend


def _check_backend_requirements(
    inferscale: InferScaleConfig,
    answer_backend: str,
    vector_backend: str,
    top_k: int,
    context_window: int,
    where: str,
) -> None:
    beam_width = inferscale.index.beam_width
    if vector_backend == "jasper" and max(beam_width, top_k) > MAX_JASPER_BEAM_WIDTH:
        raise ConfigError(
            f"{where}: effective Jasper beam width must be <= {MAX_JASPER_BEAM_WIDTH}; "
            f"got max({beam_width}, {top_k})."
        )
    if answer_backend == PROMPT_INJECTION_BACKEND and not inferscale.engine.enable_prefix_caching:
        raise ConfigError(
            f"{where}: answer_backend {PROMPT_INJECTION_BACKEND} requires "
            "inferscale.engine.enable_prefix_caching."
        )
    if answer_backend == LMCACHE_BACKEND and context_window != 0:
        raise ConfigError(
            f"{where}: answer_backend {LMCACHE_BACKEND} stores each fact alone, so "
            "context_window must be 0."
        )
    if answer_backend == PROMPT_INJECTION_BACKEND and context_window != 0:
        raise ConfigError(
            f"{where}: answer_backend {PROMPT_INJECTION_BACKEND} sends only the retrieved "
            "facts, so context_window must be 0."
        )


def _lmcache_settings(
    settings: Mapping[str, Any],
    answer_backend: str,
    sweeps: Mapping[str, MemorySweep],
    where: str,
) -> LMCacheConfig | None:
    if settings.get("lmcache") is not None:
        return parse_lmcache_config(section_of(settings, "lmcache", where), f"{where}.lmcache")
    if answer_backend == LMCACHE_BACKEND or sweeps_use_backend(sweeps.values(), LMCACHE_BACKEND):
        raise ConfigError(
            f"{where}.lmcache is required when answer_backend or a sweep variant is "
            f"{LMCACHE_BACKEND!r}."
        )
    return None


def _extraction_endpoint(
    settings: Mapping[str, Any], extraction: ExtractionConfig, where: str
) -> str:
    mem0_section = section_of(settings, "mem0", where, required=False)
    reject_unknown_keys(mem0_section, MEM0_KEYS, f"{where}.mem0")
    base_url = optional(
        mem0_section, "llm_base_url", str, DEFAULT_EXTRACTION_LLM_BASE_URL, f"{where}.mem0"
    )
    endpoint_port = urlsplit(base_url).port
    if endpoint_port is not None and endpoint_port != extraction.port:
        raise ConfigError(
            f"{where}: mem0.llm_base_url port {endpoint_port} must equal extraction.port "
            f"{extraction.port}; the fact-catalog identity records the extraction endpoint."
        )
    return base_url


def _dataset_path(settings: Mapping[str, Any], runtime: RuntimeConfig, where: str) -> Path:
    dataset_path = optional(settings, "dataset_path", str, DEFAULT_DATASET_PATH, where)
    return expand_path(dataset_path, root=runtime.root)


def _results_dir(settings: Mapping[str, Any], runtime: RuntimeConfig, where: str) -> Path:
    results_dir = optional(settings, "results_dir", str, None, where)
    if results_dir:
        return expand_path(results_dir, root=runtime.root)
    return runtime.layout.results_root


def _run_id(settings: Mapping[str, Any], where: str) -> str:
    run_id = optional(settings, "run_id", str, None, where)
    if run_id is None:
        return stamp_now()
    if not _is_directory_name(run_id):
        raise ConfigError(f"{where}.run_id must be a non-empty directory name.")
    return run_id


def _is_directory_name(name: str) -> bool:
    has_separator = "/" in name or "\\" in name
    return bool(name.strip()) and name not in {".", ".."} and not has_separator
