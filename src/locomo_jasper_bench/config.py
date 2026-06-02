from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


DEFAULT_MODEL = "shuyuej/Llama-3.3-70B-Instruct-GPTQ"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY = "token-abc123"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

Mode = Literal["baseline", "plugin", "evaluate-only"]
ContextMode = Literal["full", "retrieval"]
VectorBackend = Literal["jasper", "numpy"]
DistanceMetric = Literal["ip", "l2"]
EmbeddingProvider = Literal["openai", "hash"]


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("JSON value must be an object")
    return parsed


@dataclass(slots=True)
class BenchmarkConfig:
    mode: Mode = "baseline"
    dataset_path: Path = Path("data/locomo10.json")
    predictions_path: Path | None = None
    results_dir: Path = Path("results")
    run_id: str = field(default_factory=default_run_id)

    model: str = DEFAULT_MODEL
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: str = DEFAULT_VLLM_API_KEY
    llm_extra_body: dict[str, Any] = field(default_factory=dict)

    judge_model: str = DEFAULT_MODEL
    judge_base_url: str = DEFAULT_LLM_BASE_URL
    judge_api_key: str = DEFAULT_VLLM_API_KEY
    judge_extra_body: dict[str, Any] = field(default_factory=dict)

    embedding_provider: EmbeddingProvider = "openai"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_batch_size: int = 64
    hash_embedding_dim: int = 1536

    context_mode: ContextMode = "full"
    vector_backend: VectorBackend = "jasper"
    vector_distance: DistanceMetric = "ip"
    normalize_embeddings: bool = True
    top_k: int = 20
    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.2
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    temperature: float = 0.0
    top_p: float = 1.0
    max_answer_tokens: int = 512
    max_judge_tokens: int = 256
    stream: bool = False
    skip_judge: bool = False

    max_samples: int | None = None
    max_questions: int | None = None
    vllm_command: str | None = None
    log_every: int = 5

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("dataset_path", "predictions_path", "results_dir"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        prog="locomo-jasper-bench",
        description="Run LoCoMo through baseline and plugin vLLM servers for accuracy and latency comparisons.",
    )
    parser.add_argument("--mode", choices=["baseline", "plugin", "evaluate-only"], default="baseline")
    parser.add_argument("--dataset", dest="dataset_path", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--predictions", dest="predictions_path", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id", default=default_run_id())

    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--llm-base-url", default=os.environ.get("VLLM_BASE_URL", DEFAULT_LLM_BASE_URL))
    parser.add_argument("--llm-api-key", default=os.environ.get("VLLM_API_KEY", DEFAULT_VLLM_API_KEY))
    parser.add_argument("--llm-extra-body-json", dest="llm_extra_body", type=_json_object, default={})

    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", DEFAULT_LLM_BASE_URL))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", DEFAULT_VLLM_API_KEY))
    parser.add_argument("--judge-extra-body-json", dest="judge_extra_body", type=_json_object, default={})

    parser.add_argument("--embedding-provider", choices=["openai", "hash"], default="openai")
    parser.add_argument("--embedding-model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--embedding-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--hash-embedding-dim", type=int, default=1536)

    parser.add_argument("--context-mode", choices=["full", "retrieval"], default="full")
    parser.add_argument("--vector-backend", choices=["jasper", "numpy"], default="jasper")
    parser.add_argument("--vector-distance", choices=["ip", "l2"], default="ip")
    parser.add_argument("--no-normalize-embeddings", action="store_false", dest="normalize_embeddings")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--jasper-n-neighbors", type=int, default=64)
    parser.add_argument("--jasper-alpha", type=float, default=1.2)
    parser.add_argument("--jasper-workspace-budget", default="10GB")
    parser.add_argument("--jasper-beam-width", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--max-judge-tokens", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")

    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--vllm-command")
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("LOCOMO_LOG_EVERY", "5")))

    ns = parser.parse_args(argv)
    if ns.mode == "evaluate-only" and ns.predictions_path is None:
        parser.error("--predictions is required with --mode evaluate-only")
    return BenchmarkConfig(**vars(ns))
