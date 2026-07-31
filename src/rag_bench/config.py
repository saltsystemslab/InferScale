from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from locomo_jasper_bench.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_API_KEY,
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_MAX_JUDGE_TOKENS,
    DEFAULT_MODEL,
    MAX_JASPER_BEAM_WIDTH,
    default_run_id,
    env_flag,
    resolve_answer_model,
)
from locomo_jasper_bench.runtime_paths import (
    default_embedding_cache_dir,
    default_results_root,
)

AnswerBackend = Literal["vllm-kv", "vllm-prefix"]
JudgeProvider = Literal["vllm", "none"]

DEFAULT_DATASET = "multihoprag"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CONTEXT_WINDOW = 5
DEFAULT_TOP_K = 15
DEFAULT_MAX_ANSWER_TOKENS = 64
DEFAULT_EMBED_BATCH_SIZE = 128
# Parse-time slack for the scaffold and the templated question on top of the
# retrieved chunks; the answer path still enforces the exact budget per query.
MEMORY_BUDGET_MARGIN_TOKENS = 512


def _env_or_default(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _arg_present(argv: list[str], option: str) -> bool:
    return option in argv or any(value.startswith(f"{option}=") for value in argv)


@dataclass(slots=True)
class RagBenchConfig:
    dataset_name: str = DEFAULT_DATASET
    data_dir: Path | None = None
    results_dir: Path = field(default_factory=default_results_root)
    run_id: str = field(default_factory=default_run_id)

    model: str = DEFAULT_MODEL
    answer_backend: AnswerBackend = "vllm-kv"

    chunk_size: int = DEFAULT_CHUNK_SIZE
    context_window: int = DEFAULT_CONTEXT_WINDOW
    top_k: int = DEFAULT_TOP_K

    judge_provider: JudgeProvider = DEFAULT_JUDGE_PROVIDER
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL
    judge_api_key: str | None = DEFAULT_JUDGE_API_KEY

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = field(default_factory=default_embedding_cache_dir)
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE

    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    temperature: float = 0.0
    top_p: float = 1.0
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
    max_judge_tokens: int = DEFAULT_MAX_JUDGE_TOKENS

    kv_connector_module: str = "locomo_jasper_bench.kv.gpu_connector"
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
    estimate_only: bool = False
    preembed_only: bool = False
    precompute_kv_only: bool = False
    skip_judge: bool = False
    judge_only: bool = False
    rejudge: bool = False

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = Path("data") / self.dataset_name

    def result_mode(self) -> str:
        return "rag-kv" if self.answer_backend == "vllm-kv" else "rag-prefix"

    @property
    def jasper_effective_beam_width(self) -> int:
        return max(self.jasper_beam_width, self.top_k)

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id

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


def parse_args(argv: list[str] | None = None) -> RagBenchConfig:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="rag-jasper-bench",
        description=(
            "Run standalone RAG benchmarks (MultiHop-RAG first) with Jasper retrieval "
            "and InferScale chunked-RoPE KV injection, without Mem0 extraction."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--dataset-name",
        default=os.environ.get("RAG_DATASET", DEFAULT_DATASET),
        help="Registered RAG dataset name (currently: multihoprag).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir_from_env(),
        help="Dataset directory (default: data/<dataset-name>).",
    )
    parser.add_argument("--results-dir", type=Path, default=default_results_root())
    parser.add_argument("--run-id", default=default_run_id())

    parser.add_argument(
        "--model",
        "--answer-model",
        dest="model",
        default=os.environ.get("RAG_MODEL")
        or os.environ.get("LOCOMO_VLLM_MODEL", DEFAULT_MODEL),
        help=(
            "Answer model HF id, local path, or configured alias. "
            "Built-in aliases: llama, mistral, qwen, qwen3-14b."
        ),
    )
    parser.add_argument(
        "--answer-backend",
        choices=["vllm-kv", "vllm-prefix"],
        default=os.environ.get("RAG_ANSWER_BACKEND", "vllm-kv"),
        help="Inject retrieved chunks through in-process vLLM KV or a normal prompt prefix.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("RAG_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        help="Tokens per corpus chunk (default:1024).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=int(os.environ.get("RAG_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))),
        help=(
            "Number of same-document chunks immediately preceding each chunk used as an "
            "encoding-only prefix whose KV is discarded (vllm-kv only)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("RAG_TOP_K", str(DEFAULT_TOP_K))),
        help="Number of chunks to retrieve per query (default: 15).",
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
        "--embedding-model",
        default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument("--embedding-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=default_embedding_cache_dir())
    parser.add_argument("--no-embedding-cache", action="store_false", dest="embedding_cache_enabled")
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=int(os.environ.get("RAG_EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE))),
    )

    parser.add_argument("--jasper-n-neighbors", type=int, default=64)
    parser.add_argument("--jasper-alpha", type=float, default=1.0)
    parser.add_argument("--jasper-workspace-budget", default="10GB")
    parser.add_argument("--jasper-beam-width", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=int(os.environ.get("RAG_MAX_ANSWER_TOKENS", str(DEFAULT_MAX_ANSWER_TOKENS))),
        help="Generation cap; MultiHop-RAG gold answers are short phrases.",
    )
    parser.add_argument("--max-judge-tokens", type=int, default=DEFAULT_MAX_JUDGE_TOKENS)

    parser.add_argument(
        "--kv-connector-module",
        default=os.environ.get("LOCOMO_KV_CONNECTOR_MODULE", "locomo_jasper_bench.kv.gpu_connector"),
        help="Import path for the GPU MemoryKVConnector module used by in-process vLLM.",
    )
    parser.add_argument(
        "--kv-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("RAG_KV_GPU_MEMORY_UTILIZATION", "0.40")),
    )
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=int(os.environ.get("LOCOMO_KV_BLOCK_SIZE", "16")),
        help="vLLM KV-cache block size. The passages footer protects chunks from its partial tail.",
    )
    parser.add_argument(
        "--kv-max-model-len",
        type=int,
        default=int(os.environ.get("RAG_KV_MAX_MODEL_LEN", "32768")),
    )
    parser.add_argument(
        "--kv-max-position",
        type=int,
        default=int(os.environ.get("RAG_KV_MAX_POSITION", "32768")),
    )
    parser.add_argument("--kv-dtype", default=os.environ.get("LOCOMO_KV_DTYPE", "bfloat16"))
    parser.add_argument("--kv-device", default=os.environ.get("LOCOMO_KV_DEVICE", "cuda:0"))
    parser.add_argument(
        "--kv-prefix-caching",
        action=argparse.BooleanOptionalAction,
        dest="kv_enable_prefix_caching",
        default=env_flag("LOCOMO_KV_ENABLE_PREFIX_CACHING", True),
        help="vLLM automatic prefix caching (vllm-prefix requires it).",
    )
    parser.add_argument(
        "--kv-chunk-cache-root",
        type=Path,
        default=_optional_path_env("RAG_KV_CHUNK_CACHE_ROOT"),
        help=(
            "Root of the per-chunk KV precompute cache that the cpu store loads from "
            "(default: <cache root>/rag-kv-chunks)."
        ),
    )

    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("RAG_LOG_EVERY", "25")))
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help=(
            "Print corpus, chunk, embedding, and projected KV cache sizes for this "
            "configuration, then exit without any GPU or network work."
        ),
    )
    parser.add_argument(
        "--preembed-only",
        action="store_true",
        help="Precompute all chunk and query embeddings into the embedding cache, then exit.",
    )
    parser.add_argument(
        "--precompute-kv-only",
        action="store_true",
        help=(
            "Pre-encode every corpus chunk's KV (with its context-window prefix) into "
            "the per-chunk disk cache for this model, then exit."
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
    if ns.top_k < 1:
        parser.error("--top-k must be >= 1.")
    if ns.chunk_size < 1:
        parser.error("--chunk-size must be >= 1.")
    if ns.context_window < 0:
        parser.error("--context-window must be >= 0.")
    if ns.embed_batch_size < 1:
        parser.error("--embed-batch-size must be >= 1.")
    if ns.jasper_beam_width < 1:
        parser.error("--jasper-beam-width must be >= 1.")
    if max(ns.jasper_beam_width, ns.top_k) > MAX_JASPER_BEAM_WIDTH:
        parser.error(
            "Effective Jasper beam width must be <= "
            f"{MAX_JASPER_BEAM_WIDTH}; got max({ns.jasper_beam_width}, {ns.top_k})."
        )
    if ns.kv_block_size < 1:
        parser.error("--kv-block-size must be >= 1.")
    memory_budget = min(ns.kv_max_position, ns.kv_max_model_len - ns.max_answer_tokens)
    if ns.top_k * ns.chunk_size + MEMORY_BUDGET_MARGIN_TOKENS > memory_budget:
        parser.error(
            f"top_k x chunk_size + {MEMORY_BUDGET_MARGIN_TOKENS} scaffold/query margin "
            f"({ns.top_k} x {ns.chunk_size} + {MEMORY_BUDGET_MARGIN_TOKENS}) exceeds the "
            f"memory budget min(--kv-max-position, --kv-max-model-len - --max-answer-tokens) "
            f"= {memory_budget}. Lower --top-k or --chunk-size, or raise the limits."
        )
    if ns.answer_backend == "vllm-prefix" and not ns.kv_enable_prefix_caching:
        parser.error(
            "--answer-backend vllm-prefix requires prefix caching; pass --kv-prefix-caching "
            "or unset LOCOMO_KV_ENABLE_PREFIX_CACHING."
        )
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
    return RagBenchConfig(**vars(ns))


def _default_data_dir_from_env() -> Path | None:
    for name in ("RAG_DATA_DIR", "MULTIHOP_RAG_DATA_DIR"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    return None


def _optional_path_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


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
