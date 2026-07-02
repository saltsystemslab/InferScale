from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_JUDGE_MODEL = "google/gemma-2-9b-it"
DEFAULT_JUDGE_BASE_URL = "http://localhost:8000/v1"
DEFAULT_JUDGE_API_KEY = "token-abc123"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

DistanceMetric = Literal["ip", "l2"]
AnswerBackend = Literal["vllm-kv", "vllm-prefix"]
VectorBackend = Literal["jasper", "qdrant"]
MemoryOrder = Literal["retrieval", "turn-index", "rank-zigzag", "retrieval-reversed"]


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_results_dir() -> Path:
    if "BENCHMARK_RESULTS_ROOT" in os.environ:
        return Path(os.environ["BENCHMARK_RESULTS_ROOT"])
    if "SCRATCH_ROOT" in os.environ:
        return Path(os.environ["SCRATCH_ROOT"]) / "results"
    return Path("results")


def default_embedding_cache_dir() -> Path:
    if "BENCHMARK_CACHE_ROOT" in os.environ:
        cache_root = Path(os.environ["BENCHMARK_CACHE_ROOT"])
    elif "SCRATCH_ROOT" in os.environ:
        cache_root = Path(os.environ["SCRATCH_ROOT"]) / "cache"
    else:
        cache_root = Path(".cache")
    return cache_root / "embeddings"


@dataclass(slots=True)
class BenchmarkConfig:
    dataset_path: Path = Path("data/locomo10.json")
    results_dir: Path = field(default_factory=default_results_dir)
    run_id: str = field(default_factory=default_run_id)

    model: str = DEFAULT_MODEL
    answer_backend: AnswerBackend = "vllm-kv"
    memory_order: MemoryOrder = "retrieval"

    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_base_url: str = DEFAULT_JUDGE_BASE_URL
    judge_api_key: str = DEFAULT_JUDGE_API_KEY

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = field(default_factory=default_embedding_cache_dir)

    vector_backend: VectorBackend = "jasper"
    vector_distance: DistanceMetric = "ip"
    top_k: int = 50
    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    temperature: float = 0.0
    top_p: float = 1.0
    max_answer_tokens: int = 512
    max_judge_tokens: int = 4

    kv_connector_module: str = "locomo_jasper_bench.kv.gpu_connector"
    context_window: int = 3
    kv_gpu_memory_utilization: float = 0.52
    kv_max_model_len: int = 32768
    kv_max_position: int = 32768
    kv_dtype: str = "bfloat16"
    kv_device: str = "cuda:0"

    max_samples: int | None = None
    max_questions: int | None = None
    log_every: int = 5
    preembed_only: bool = False
    skip_judge: bool = False
    judge_only: bool = False

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("dataset_path", "results_dir", "embedding_cache_dir"):
            if data[key] is not None:
                data[key] = str(data[key])
        for key in ("judge_api_key", "embedding_api_key"):
            if data.get(key):
                data[key] = "<redacted>"
        return data

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    # Single source of truth for CLI defaults so they cannot drift from the dataclass.
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(
        prog="locomo-jasper-bench",
        description="Run LoCoMo KV-cache benchmarks with Mem0 retrieval backed by Jasper or Qdrant.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", dest="dataset_path", type=Path, default=defaults.dataset_path)
    parser.add_argument("--results-dir", type=Path, default=defaults.results_dir)
    parser.add_argument("--run-id", default=defaults.run_id)

    parser.add_argument("--model", default=os.environ.get("LOCOMO_VLLM_MODEL", defaults.model))
    parser.add_argument(
        "--answer-backend",
        choices=["vllm-kv", "vllm-prefix"],
        default=os.environ.get("LOCOMO_ANSWER_BACKEND", defaults.answer_backend),
        help="Use in-process vLLM KV injection or the same-token prefix prompt baseline.",
    )
    parser.add_argument(
        "--memory-order",
        choices=["retrieval", "turn-index", "rank-zigzag", "retrieval-reversed"],
        default=None,
        help=(
            "Memory injection order for vllm-kv and vllm-prefix: retrieval rank, chronological "
            "LoCoMo turn order, alternating retrieval-rank ends, or reversed retrieval rank."
        ),
    )
    parser.add_argument(
        "--prefix-memory-order",
        dest="legacy_prefix_memory_order",
        choices=["retrieval", "turn-index", "rank-zigzag", "retrieval-reversed"],
        default=None,
        help="Deprecated alias for --memory-order.",
    )

    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", defaults.judge_model))
    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", defaults.judge_base_url))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", defaults.judge_api_key))

    parser.add_argument("--embedding-model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", defaults.embedding_model))
    parser.add_argument("--embedding-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=defaults.embedding_cache_dir)
    parser.add_argument("--no-embedding-cache", action="store_false", dest="embedding_cache_enabled")

    parser.add_argument("--vector-backend", choices=["jasper", "qdrant"], default=defaults.vector_backend)
    parser.add_argument("--vector-distance", choices=["ip", "l2"], default=defaults.vector_distance)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--jasper-n-neighbors", type=int, default=defaults.jasper_n_neighbors)
    parser.add_argument("--jasper-alpha", type=float, default=defaults.jasper_alpha)
    parser.add_argument("--jasper-workspace-budget", default=defaults.jasper_workspace_budget)
    parser.add_argument("--jasper-beam-width", type=int, default=defaults.jasper_beam_width)

    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--max-answer-tokens", type=int, default=defaults.max_answer_tokens)
    parser.add_argument("--max-judge-tokens", type=int, default=defaults.max_judge_tokens)

    parser.add_argument(
        "--kv-connector-module",
        default=os.environ.get("LOCOMO_KV_CONNECTOR_MODULE", defaults.kv_connector_module),
        help="Import path for the GPU MemoryKVConnector module used by in-process vLLM.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=int(os.environ.get("LOCOMO_KV_CONTEXT_WINDOW", defaults.context_window)),
        help=(
            "Number of previous LoCoMo sessions to include as prefix context when "
            "pre-RoPE encoding each selected KV memory turn. 0 encodes each turn in isolation."
        ),
    )
    parser.add_argument(
        "--kv-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("LOCOMO_KV_GPU_MEMORY_UTILIZATION", defaults.kv_gpu_memory_utilization)),
    )
    parser.add_argument(
        "--kv-max-model-len",
        type=int,
        default=int(os.environ.get("LOCOMO_KV_MAX_MODEL_LEN", defaults.kv_max_model_len)),
    )
    parser.add_argument(
        "--kv-max-position",
        type=int,
        default=int(os.environ.get("LOCOMO_KV_MAX_POSITION", defaults.kv_max_position)),
        help="Maximum RoPE virtual position for top-k memory composition.",
    )
    parser.add_argument("--kv-dtype", default=os.environ.get("LOCOMO_KV_DTYPE", defaults.kv_dtype))
    parser.add_argument("--kv-device", default=os.environ.get("LOCOMO_KV_DEVICE", defaults.kv_device))

    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("LOCOMO_LOG_EVERY", defaults.log_every)))
    parser.add_argument(
        "--preembed-only",
        action="store_true",
        help="Precompute LoCoMo turn and question embeddings into the cache, then exit.",
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

    ns = parser.parse_args(argv)
    if ns.answer_backend not in {"vllm-kv", "vllm-prefix"}:
        parser.error("--answer-backend must be vllm-kv or vllm-prefix.")
    if ns.memory_order and ns.legacy_prefix_memory_order and ns.memory_order != ns.legacy_prefix_memory_order:
        parser.error("--memory-order and --prefix-memory-order must match when both are provided.")
    ns.memory_order = (
        ns.memory_order
        or ns.legacy_prefix_memory_order
        or os.environ.get("LOCOMO_MEMORY_ORDER")
        or os.environ.get("LOCOMO_PREFIX_MEMORY_ORDER")
        or "retrieval"
    )
    del ns.legacy_prefix_memory_order
    if ns.memory_order not in {"retrieval", "turn-index", "rank-zigzag", "retrieval-reversed"}:
        parser.error("--memory-order must be retrieval, turn-index, rank-zigzag, or retrieval-reversed.")
    if ns.context_window < 0:
        parser.error("--context-window must be >= 0.")
    return BenchmarkConfig(**vars(ns))
