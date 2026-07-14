from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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
from .kv.connector_utils import (
    DEFAULT_KV_STAGING_SLOTS,
    DEFAULT_KV_STORE_BACKEND,
    KNOWN_KV_STORE_BACKENDS,
)
from .runtime_paths import default_embedding_cache_dir as runtime_default_embedding_cache_dir
from .runtime_paths import default_memory_llm_cache_dir as runtime_default_memory_llm_cache_dir
from .runtime_paths import default_results_root


DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_QWEN3_14B_MODEL = "Qwen/Qwen3-14B"
DEFAULT_MODEL = DEFAULT_LLAMA_MODEL
DEFAULT_JUDGE_PROVIDER = "vllm"
DEFAULT_JUDGE_MODEL = "Gemma-2-9B-Instruct"
DEFAULT_JUDGE_BASE_URL = "http://localhost:8000/v1"
DEFAULT_JUDGE_API_KEY = "token-abc123"
DEFAULT_MAX_JUDGE_TOKENS = 4
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MEMORY_LLM_PROVIDER = "vllm"

ANSWER_MODEL_DEFAULTS = {
    "llama": DEFAULT_LLAMA_MODEL,
    "mistral": DEFAULT_MISTRAL_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
    "qwen3-14b": DEFAULT_QWEN3_14B_MODEL,
}
ANSWER_MODEL_ENV_VARS = {
    "llama": ("LOCOMO_MODEL_LLAMA", "MODEL_LLAMA"),
    "mistral": ("LOCOMO_MODEL_MISTRAL", "MODEL_MISTRAL"),
    "qwen": ("LOCOMO_MODEL_QWEN", "MODEL_QWEN"),
    "qwen3-14b": ("LOCOMO_MODEL_QWEN3_14B", "MODEL_QWEN3_14B"),
}
# vLLM serve reasoning parsers per alias key. Reasoning models emit thinking
# tokens; the parser keeps them out of message content so mem0's JSON
# extraction parsing sees only the final answer.
ANSWER_MODEL_REASONING_PARSERS = {
    "qwen3-14b": "qwen3",
}
ANSWER_MODEL_NAME_ALIASES = {
    "llama": "llama",
    "llama3": "llama",
    "llama3.1": "llama",
    "llama-3.1": "llama",
    "llama-3.1-8b-instruct": "llama",
    "mistral": "mistral",
    "mistral-7b": "mistral",
    "mistral-7b-instruct-v0.3": "mistral",
    "qwen": "qwen",
    "qwen2.5": "qwen",
    "qwen2.5-7b": "qwen",
    "qwen2.5-7b-instruct": "qwen",
    "qwen3": "qwen3-14b",
    "qwen3-14b": "qwen3-14b",
}

AnswerBackend = Literal["vllm-kv", "vllm-prefix"]
JudgeProvider = Literal["vllm", "none"]
MemoryLlmProvider = Literal["vllm"]
VectorBackend = Literal["jasper", "qdrant"]
MEMORY_UNIT = "mem0-fact"
CONTEXT_WINDOW_UNIT = "turns"
CONTEXT_WINDOW_SEMANTICS = "fact-encoding-prefix-discard-v1"
MAX_JASPER_BEAM_WIDTH = 959


def configured_answer_models() -> dict[str, str]:
    models: dict[str, str] = {}
    for name, default in ANSWER_MODEL_DEFAULTS.items():
        models[name] = next(
            (
                value
                for env_var in ANSWER_MODEL_ENV_VARS[name]
                if (value := os.environ.get(env_var))
            ),
            default,
        )
    return models


def answer_model_aliases() -> dict[str, str]:
    configured = configured_answer_models()
    return {
        alias: configured[model_name]
        for alias, model_name in ANSWER_MODEL_NAME_ALIASES.items()
    }


def resolve_answer_model(model: str) -> str:
    stripped = model.strip()
    return answer_model_aliases().get(stripped.lower(), stripped)


def resolve_reasoning_parser(model: str) -> str | None:
    stripped = model.strip()
    alias = ANSWER_MODEL_NAME_ALIASES.get(stripped.lower())
    if alias is None:
        alias = next(
            (
                name
                for name, configured in configured_answer_models().items()
                if configured == stripped
            ),
            None,
        )
    if alias is None:
        return None
    return ANSWER_MODEL_REASONING_PARSERS.get(alias)


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_results_dir() -> Path:
    return default_results_root()


def default_embedding_cache_dir() -> Path:
    return runtime_default_embedding_cache_dir()


def default_memory_llm_cache_dir() -> Path:
    return runtime_default_memory_llm_cache_dir()


def _env_or_default(name: str, default: str) -> str:
    return os.environ.get(name) or default


def is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return is_truthy(value)


def _arg_present(argv: list[str], option: str) -> bool:
    return option in argv or any(value.startswith(f"{option}=") for value in argv)


@dataclass(slots=True)
class BenchmarkConfig:
    dataset_path: Path = Path("data/locomo10.json")
    results_dir: Path = field(default_factory=default_results_dir)
    run_id: str = field(default_factory=default_run_id)

    model: str = DEFAULT_MODEL
    answer_backend: AnswerBackend = "vllm-kv"

    judge_provider: JudgeProvider = DEFAULT_JUDGE_PROVIDER
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL
    judge_api_key: str | None = DEFAULT_JUDGE_API_KEY
    with_evidence: bool = False

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = field(default_factory=default_embedding_cache_dir)

    memory_llm_provider: MemoryLlmProvider = DEFAULT_MEMORY_LLM_PROVIDER
    # Mem0 fact extraction always uses the answer model; see __post_init__.
    memory_llm_model: str | None = None
    memory_llm_base_url: str | None = None
    memory_llm_api_key: str | None = None
    memory_llm_cache_dir: Path = field(default_factory=default_memory_llm_cache_dir)
    memory_unit: Literal["mem0-fact"] = field(default=MEMORY_UNIT, init=False)
    mem0_infer: Literal[True] = field(default=True, init=False)
    memory_benchmarks_repository: str = field(
        default=MEMORY_BENCHMARKS_REPOSITORY,
        init=False,
    )
    memory_benchmarks_commit: str = field(default=MEMORY_BENCHMARKS_COMMIT, init=False)
    mem0ai_version: str = field(default=MEM0AI_VERSION, init=False)
    memory_extraction_response_protocol: str = field(
        default=MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
        init=False,
    )
    memory_extraction_max_model_len: int = field(
        default=MEMORY_EXTRACTION_MAX_MODEL_LEN,
        init=False,
    )
    memory_extraction_max_tokens: int = field(
        default=MEMORY_EXTRACTION_MAX_TOKENS,
        init=False,
    )
    memory_extraction_max_facts: int = field(
        default=MEMORY_EXTRACTION_MAX_FACTS,
        init=False,
    )
    memory_extraction_max_text_chars: int = field(
        default=MEMORY_EXTRACTION_MAX_TEXT_CHARS,
        init=False,
    )
    memory_ingestion_protocol: str = field(default=MEMORY_INGESTION_PROTOCOL, init=False)
    answer_prompt_protocol: str = field(default=ANSWER_PROMPT_PROTOCOL, init=False)
    judge_prompt_protocol: str = field(default=JUDGE_PROMPT_PROTOCOL, init=False)

    vector_backend: VectorBackend = "jasper"
    top_k: int = 50
    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    temperature: float = 0.0
    top_p: float = 1.0
    max_answer_tokens: int = 512
    max_judge_tokens: int = DEFAULT_MAX_JUDGE_TOKENS

    kv_connector_module: str = "locomo_jasper_bench.kv.gpu_connector"
    context_window: int = 0
    context_window_unit: Literal["turns"] = field(default=CONTEXT_WINDOW_UNIT, init=False)
    context_window_semantics: Literal["fact-encoding-prefix-discard-v1"] = field(
        default=CONTEXT_WINDOW_SEMANTICS,
        init=False,
    )
    kv_gpu_memory_utilization: float = 0.30
    kv_block_size: int = 16
    kv_max_model_len: int = 32768
    kv_max_position: int = 32768
    kv_dtype: str = "bfloat16"
    kv_device: str = "cuda:0"
    kv_enable_prefix_caching: bool = True
    kv_store_backend: str = DEFAULT_KV_STORE_BACKEND
    kv_staging_slots: int = DEFAULT_KV_STAGING_SLOTS

    max_samples: int | None = None
    max_questions: int | None = None
    preembed_workers: int = 4
    log_every: int = 5
    check_catalogs: bool = False
    preembed_only: bool = False
    skip_judge: bool = False
    judge_only: bool = False
    rejudge: bool = False

    def __post_init__(self) -> None:
        if self.memory_llm_model is None:
            self.memory_llm_model = self.model
        elif self.memory_llm_model != self.model:
            raise ValueError(
                "Mem0 fact extraction always uses the answer model; "
                f"memory_llm_model={self.memory_llm_model!r} conflicts with model={self.model!r}."
            )

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        data["jasper_effective_beam_width"] = self.jasper_effective_beam_width
        for key in ("dataset_path", "results_dir", "embedding_cache_dir", "memory_llm_cache_dir"):
            if data[key] is not None:
                data[key] = str(data[key])
        for key in ("judge_api_key", "embedding_api_key", "memory_llm_api_key"):
            if data.get(key):
                data[key] = "<redacted>"
        return data

    @property
    def jasper_effective_beam_width(self) -> int | None:
        if self.vector_backend != "jasper":
            return None
        return max(self.jasper_beam_width, self.top_k)

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="locomo-jasper-bench",
        description="Run LoCoMo KV-cache benchmarks with Mem0 retrieval backed by Jasper or Qdrant.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", dest="dataset_path", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--run-id", default=default_run_id())

    parser.add_argument(
        "--model",
        "--answer-model",
        dest="model",
        default=os.environ.get("LOCOMO_VLLM_MODEL", DEFAULT_MODEL),
        help=(
            "Answer model HF id, local path, or configured alias. "
            "Built-in aliases: llama, mistral, qwen, qwen3-14b."
        ),
    )
    parser.add_argument(
        "--answer-backend",
        choices=["vllm-kv", "vllm-prefix"],
        default=os.environ.get("LOCOMO_ANSWER_BACKEND", "vllm-kv"),
        help="Inject retrieved Mem0 facts through in-process vLLM KV or a normal prompt prefix.",
    )

    parser.add_argument(
        "--judge",
        dest="judge_provider",
        choices=["vllm", "none"],
        default=None,
        help="Judge provider to use: local OpenAI-compatible vLLM or none.",
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key")
    parser.add_argument(
        "--with-evidence",
        action="store_true",
        dest="with_evidence",
        help=(
            "Pass the gold evidence turns to the judge (upstream --with-evidence). "
            "Off by default to match the upstream scoring methodology."
        ),
    )

    parser.add_argument("--embedding-model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--embedding-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=default_embedding_cache_dir())
    parser.add_argument("--no-embedding-cache", action="store_false", dest="embedding_cache_enabled")
    parser.set_defaults(memory_llm_provider=DEFAULT_MEMORY_LLM_PROVIDER)
    parser.add_argument(
        "--memory-llm-base-url",
        default=os.environ.get("MEM0_LLM_BASE_URL"),
        help=(
            "vLLM OpenAI-compatible base URL for Mem0 extraction. When unset, mem0 falls "
            "back to VLLM_BASE_URL or http://localhost:8000/v1."
        ),
    )
    parser.add_argument(
        "--memory-llm-api-key",
        default=os.environ.get("MEM0_LLM_API_KEY"),
        help="Optional API key for the Mem0 extraction vLLM server.",
    )
    parser.add_argument(
        "--memory-llm-cache-dir",
        type=Path,
        default=default_memory_llm_cache_dir(),
        help="Directory for deterministic Mem0 inference response caching.",
    )

    parser.add_argument("--vector-backend", choices=["jasper", "qdrant"], default="jasper")
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of inferred Mem0 fact records to retrieve. The primary benchmark setting is 50.",
    )
    parser.add_argument("--jasper-n-neighbors", type=int, default=64)
    parser.add_argument("--jasper-alpha", type=float, default=1.0)
    parser.add_argument("--jasper-workspace-budget", default="10GB")
    parser.add_argument("--jasper-beam-width", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--max-judge-tokens", type=int, default=DEFAULT_MAX_JUDGE_TOKENS)

    parser.add_argument(
        "--kv-connector-module",
        default=os.environ.get("LOCOMO_KV_CONNECTOR_MODULE", "locomo_jasper_bench.kv.gpu_connector"),
        help="Import path for the GPU MemoryKVConnector module used by in-process vLLM.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=int(os.environ.get("LOCOMO_KV_CONTEXT_WINDOW", "0")),
        help=(
            "Window of conversation turns immediately preceding each retrieved Mem0 fact's "
            "source turn. vllm-kv uses the catalog facts extracted from those turns as an "
            "encoding-only prefix and discards the prefix KV; vllm-prefix renders the "
            "deduplicated raw turns in the prompt."
        ),
    )
    parser.add_argument(
        "--kv-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("LOCOMO_KV_GPU_MEMORY_UTILIZATION", "0.30")),
    )
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=int(os.environ.get("LOCOMO_KV_BLOCK_SIZE", "16")),
        help="vLLM KV-cache block size. The memory footer protects facts from its partial tail.",
    )
    parser.add_argument("--kv-dtype", default=os.environ.get("LOCOMO_KV_DTYPE", "bfloat16"))
    parser.add_argument("--kv-device", default=os.environ.get("LOCOMO_KV_DEVICE", "cuda:0"))
    parser.add_argument(
        "--kv-prefix-caching",
        action=argparse.BooleanOptionalAction,
        dest="kv_enable_prefix_caching",
        default=env_flag("LOCOMO_KV_ENABLE_PREFIX_CACHING", True),
        help=(
            "vLLM automatic prefix caching (vllm-prefix requires it). The CLI "
            "overrides the LOCOMO_KV_ENABLE_PREFIX_CACHING env default in both directions."
        ),
    )
    parser.add_argument(
        "--kv-store-backend",
        choices=list(KNOWN_KV_STORE_BACKENDS),
        default=os.environ.get("LOCOMO_KV_STORE_BACKEND", DEFAULT_KV_STORE_BACKEND),
        help="Where pre-encoded KV embeddings live: GPU HBM, or pinned host RAM streamed over PCIe.",
    )
    parser.add_argument(
        "--kv-staging-slots",
        type=int,
        default=int(os.environ.get("LOCOMO_KV_STAGING_SLOTS", str(DEFAULT_KV_STAGING_SLOTS))),
        help="GPU staging buffers kept in flight by the cpu-pinned KV store.",
    )

    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument(
        "--preembed-workers",
        type=int,
        default=int(os.environ.get("LOCOMO_PREEMBED_WORKERS", "4")),
        help="Concurrent conversations used by --preembed-only; turns within each conversation remain serial.",
    )
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("LOCOMO_LOG_EVERY", "5")))
    parser.add_argument(
        "--check-catalogs",
        action="store_true",
        help=(
            "Verify that every sample's Mem0 fact catalog exists for this configuration's "
            "full catalog identity, then exit without running the benchmark."
        ),
    )
    parser.add_argument(
        "--preembed-only",
        action="store_true",
        help=(
            "Materialize backend-independent Mem0 fact catalogs and precompute all required "
            "embeddings, then exit."
        ),
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Generate predictions without calling the judge endpoint.",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Judge missing results in an existing run directory and regenerate summary.json.",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="With --judge-only, replace existing judge results instead of only filling missing results.",
    )

    ns = parser.parse_args(raw_argv)
    if ns.answer_backend not in {"vllm-kv", "vllm-prefix"}:
        parser.error("--answer-backend must be vllm-kv or vllm-prefix.")
    if ns.top_k < 1:
        parser.error("--top-k must be >= 1.")
    if ns.jasper_beam_width < 1:
        parser.error("--jasper-beam-width must be >= 1.")
    if ns.vector_backend == "jasper" and max(ns.jasper_beam_width, ns.top_k) > MAX_JASPER_BEAM_WIDTH:
        parser.error(
            "Effective Jasper beam width must be <= "
            f"{MAX_JASPER_BEAM_WIDTH}; got max({ns.jasper_beam_width}, {ns.top_k})."
        )
    if ns.context_window < 0:
        parser.error("--context-window must be >= 0.")
    if ns.preembed_workers < 1:
        parser.error("--preembed-workers must be >= 1.")
    if ns.kv_block_size < 1:
        parser.error("--kv-block-size must be >= 1.")
    if ns.answer_backend == "vllm-prefix" and not ns.kv_enable_prefix_caching:
        parser.error(
            "--answer-backend vllm-prefix requires prefix caching; pass --kv-prefix-caching "
            "or unset LOCOMO_KV_ENABLE_PREFIX_CACHING."
        )
    if ns.kv_staging_slots < 1:
        parser.error("--kv-staging-slots must be >= 1.")
    if ns.skip_judge and ns.judge_provider is not None:
        parser.error("--skip-judge cannot be combined with --judge.")
    try:
        ns.judge_provider = _resolve_judge_provider(ns.judge_provider, skip_judge=ns.skip_judge)
    except ValueError as exc:
        parser.error(str(exc))
    if ns.judge_provider == "none":
        ns.skip_judge = True
    if ns.rejudge and not ns.judge_only:
        parser.error("--rejudge requires --judge-only.")
    if ns.judge_only and ns.judge_provider == "none":
        parser.error("--judge-only requires --judge vllm.")
    _resolve_judge_connection(ns, explicit_argv=raw_argv)
    ns.model = resolve_answer_model(ns.model)
    return BenchmarkConfig(**vars(ns))


def _resolve_judge_provider(value: str | None, *, skip_judge: bool) -> JudgeProvider:
    if skip_judge:
        return "none"
    resolved = value or os.environ.get("JUDGE_PROVIDER") or DEFAULT_JUDGE_PROVIDER
    if resolved not in {"vllm", "none"}:
        raise ValueError(f"JUDGE_PROVIDER must be vllm or none, got {resolved!r}.")
    return resolved  # type: ignore[return-value]


def _resolve_judge_connection(ns: argparse.Namespace, *, explicit_argv: list[str]) -> None:
    explicit_model = _arg_present(explicit_argv, "--judge-model")
    explicit_base_url = _arg_present(explicit_argv, "--judge-base-url")
    explicit_api_key = _arg_present(explicit_argv, "--judge-api-key")

    if ns.judge_provider == "vllm":
        ns.judge_model = ns.judge_model if explicit_model else _env_or_default("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        ns.judge_base_url = ns.judge_base_url if explicit_base_url else _env_or_default(
            "JUDGE_BASE_URL",
            DEFAULT_JUDGE_BASE_URL,
        )
        ns.judge_api_key = ns.judge_api_key if explicit_api_key else _env_or_default(
            "JUDGE_API_KEY",
            DEFAULT_JUDGE_API_KEY,
        )
        return

    ns.judge_model = ns.judge_model or ""
    ns.judge_base_url = ns.judge_base_url
    ns.judge_api_key = ns.judge_api_key
