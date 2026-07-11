from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import ANSWER_MODEL_NAME_ALIASES, MAX_JASPER_BEAM_WIDTH, resolve_answer_model
from ..runtime_paths import (
    default_embedding_cache_dir,
    default_memory_llm_cache_dir,
    default_results_root,
)

ALL_CONDITIONS = (
    "no_memory",
    "mem0_qdrant",
    "mem0_jasper",
    "kv_injection",
)

# Which vector backend a condition retrieves with. no_memory does no retrieval;
# kv_injection performs the same Jasper top-k search as mem0_jasper and injects
# the retrieved chunks' KV instead of their text.
CONDITION_VECTOR_BACKENDS: dict[str, str | None] = {
    "no_memory": None,
    "mem0_qdrant": "qdrant",
    "mem0_jasper": "jasper",
    "kv_injection": "jasper",
}


def condition_vector_backend(condition: str) -> str | None:
    return CONDITION_VECTOR_BACKENDS[condition]


DEFAULT_USER_COUNTS = (10, 25, 50, 100)
DEFAULT_USER_COUNTS_TEXT = ",".join(str(count) for count in DEFAULT_USER_COUNTS)


@dataclass(slots=True)
class ThroughputConfig:
    model: str
    model_label: str
    results_dir: Path
    run_id: str
    dataset_path: Path = Path("data/locomo10.json")
    conditions: tuple[str, ...] = ALL_CONDITIONS
    user_counts: tuple[int, ...] = DEFAULT_USER_COUNTS
    requests_per_user: int = 2
    max_output_tokens: int = 50
    warmup_batches: int = 2
    top_k: int = 10
    seed: int = 42
    kv_gpu_memory_utilization: float = 0.52
    kv_max_model_len: int = 32768
    kv_max_position: int = 32768
    kv_dtype: str = "bfloat16"
    kv_device: str = "cuda:0"
    kv_block_size: int = 16
    kv_connector_module: str = "locomo_jasper_bench.kv.gpu_connector"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = None  # type: ignore[assignment]
    memory_llm_provider: str = "vllm"
    # Mem0 fact extraction always uses the answer model; see __post_init__.
    memory_llm_model: str | None = None
    memory_llm_base_url: str | None = None
    memory_llm_cache_dir: Path = None  # type: ignore[assignment]
    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    def __post_init__(self) -> None:
        if self.embedding_cache_dir is None:
            self.embedding_cache_dir = default_embedding_cache_dir()
        if self.memory_llm_cache_dir is None:
            self.memory_llm_cache_dir = default_memory_llm_cache_dir()
        if self.memory_llm_model is None:
            self.memory_llm_model = self.model
        elif self.memory_llm_model != self.model:
            raise ValueError(
                "Mem0 fact extraction always uses the answer model; "
                f"memory_llm_model={self.memory_llm_model!r} conflicts with model={self.model!r}."
            )

    @property
    def run_dir(self) -> Path:
        return self.results_dir / "throughput" / self.run_id

    def to_jsonable(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        data = asdict(self)
        for key in ("results_dir", "dataset_path", "embedding_cache_dir", "memory_llm_cache_dir"):
            data[key] = str(data[key])
        data["user_counts"] = list(self.user_counts)
        data["conditions"] = list(self.conditions)
        if redact_secrets and data.get("embedding_api_key"):
            data["embedding_api_key"] = "<redacted>"
        return data

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ThroughputConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Throughput config must be a JSON object: {path}")
        data = dict(raw)
        for key in ("results_dir", "dataset_path", "embedding_cache_dir", "memory_llm_cache_dir"):
            data[key] = Path(data[key])
        data["conditions"] = tuple(data["conditions"])
        data["user_counts"] = tuple(int(count) for count in data["user_counts"])
        if data.get("embedding_api_key") == "<redacted>":
            data["embedding_api_key"] = (
                os.environ.get("LOCOMO_THROUGHPUT_EMBEDDING_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
        return cls(**data)


def parse_user_counts(value: str | Iterable[str]) -> tuple[int, ...]:
    raw_parts: list[str] = []
    values = [value] if isinstance(value, str) else list(value)
    for item in values:
        raw_parts.extend(part.strip() for part in re.split(r"[;,\s]+", item) if part.strip())
    if not raw_parts:
        raise ValueError("The user-count list cannot be empty.")

    counts: list[int] = []
    seen: set[int] = set()
    for raw in raw_parts:
        try:
            count = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid user count {raw!r}; expected an integer.") from exc
        if count <= 0:
            raise ValueError(f"User counts must be greater than zero, got {count}.")
        if count in seen:
            raise ValueError(f"Duplicate user count: {count}")
        counts.append(count)
        seen.add(count)
    return tuple(counts)


def user_counts_text(counts: Iterable[int]) -> str:
    return ",".join(str(count) for count in counts)


def default_run_id(model_label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_slug(model_label)}-{stamp}"


def parse_args(argv: list[str] | None = None) -> tuple[ThroughputConfig, bool]:
    parser = argparse.ArgumentParser(
        prog="locomo-throughput-bench",
        description=(
            "Run multi-user throughput benchmarks for vLLM memory conditions over the "
            "LoCoMo dataset's Mem0-extracted fact catalogs."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LOCOMO_VLLM_MODEL", "llama"),
        help="Hugging Face model id, local path, or configured alias: llama, mistral, qwen, qwen3-14b.",
    )
    parser.add_argument("--results-dir", type=Path, default=default_results_root())
    parser.add_argument("--run-id")
    parser.add_argument("--dataset", dest="dataset_path", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS, default=list(ALL_CONDITIONS))
    parser.add_argument(
        "--user-counts",
        default=os.environ.get("THROUGHPUT_USER_COUNTS", DEFAULT_USER_COUNTS_TEXT),
        help="Comma-separated simulated user counts; users map to LoCoMo conversations round-robin.",
    )
    parser.add_argument("--requests-per-user", type=int, default=int(os.environ.get("THROUGHPUT_REQUESTS_PER_USER", "2")))
    parser.add_argument("--max-output-tokens", type=int, default=int(os.environ.get("THROUGHPUT_MAX_OUTPUT_TOKENS", "50")))
    parser.add_argument("--warmup-batches", type=int, default=int(os.environ.get("THROUGHPUT_WARMUP_BATCHES", "2")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("THROUGHPUT_TOP_K", "10")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("THROUGHPUT_SEED", "42")))
    parser.add_argument(
        "--gpu-memory-utilization",
        dest="kv_gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("LOCOMO_KV_GPU_MEMORY_UTILIZATION", "0.52")),
    )
    parser.add_argument("--max-model-len", dest="kv_max_model_len", type=int, default=32768)
    parser.add_argument("--kv-max-position", type=int, default=32768)
    parser.add_argument("--dtype", dest="kv_dtype", default=os.environ.get("LOCOMO_KV_DTYPE", "bfloat16"))
    parser.add_argument("--device", dest="kv_device", default=os.environ.get("LOCOMO_KV_DEVICE", "cuda:0"))
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument(
        "--kv-connector-module",
        default=os.environ.get("LOCOMO_KV_CONNECTOR_MODULE", "locomo_jasper_bench.kv.gpu_connector"),
    )
    parser.add_argument("--embedding-model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=default_embedding_cache_dir())
    parser.add_argument("--no-embedding-cache", action="store_false", dest="embedding_cache_enabled")
    parser.set_defaults(memory_llm_provider="vllm")
    parser.add_argument(
        "--memory-llm-base-url",
        default=os.environ.get("MEM0_LLM_BASE_URL"),
        help="Extraction server recorded in the fact-catalog identity (must match --preembed-only).",
    )
    parser.add_argument(
        "--memory-llm-cache-dir",
        type=Path,
        default=default_memory_llm_cache_dir(),
        help="Root directory holding the immutable fact catalogs from --preembed-only.",
    )
    parser.add_argument("--jasper-n-neighbors", type=int, default=64)
    parser.add_argument("--jasper-alpha", type=float, default=1.0)
    parser.add_argument("--jasper-workspace-budget", default="10GB")
    parser.add_argument("--jasper-beam-width", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")

    ns = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        user_counts = parse_user_counts(ns.user_counts)
        _validate_positive_options(ns)
    except ValueError as exc:
        parser.error(str(exc))

    raw_model = ns.model.strip()
    model_label = ANSWER_MODEL_NAME_ALIASES.get(raw_model.lower(), raw_model.rsplit("/", 1)[-1])
    config = ThroughputConfig(
        model=resolve_answer_model(raw_model),
        model_label=model_label,
        results_dir=ns.results_dir,
        run_id=ns.run_id or default_run_id(model_label),
        dataset_path=ns.dataset_path,
        conditions=tuple(dict.fromkeys(ns.conditions)),
        user_counts=user_counts,
        requests_per_user=ns.requests_per_user,
        max_output_tokens=ns.max_output_tokens,
        warmup_batches=ns.warmup_batches,
        top_k=ns.top_k,
        seed=ns.seed,
        kv_gpu_memory_utilization=ns.kv_gpu_memory_utilization,
        kv_max_model_len=ns.kv_max_model_len,
        kv_max_position=ns.kv_max_position,
        kv_dtype=ns.kv_dtype,
        kv_device=ns.kv_device,
        kv_block_size=ns.kv_block_size,
        kv_connector_module=ns.kv_connector_module,
        embedding_model=ns.embedding_model,
        embedding_api_key=ns.embedding_api_key,
        embedding_base_url=ns.embedding_base_url,
        embedding_cache_enabled=ns.embedding_cache_enabled,
        embedding_cache_dir=ns.embedding_cache_dir,
        memory_llm_provider=ns.memory_llm_provider,
        memory_llm_base_url=ns.memory_llm_base_url,
        memory_llm_cache_dir=ns.memory_llm_cache_dir,
        jasper_n_neighbors=ns.jasper_n_neighbors,
        jasper_alpha=ns.jasper_alpha,
        jasper_workspace_budget=ns.jasper_workspace_budget,
        jasper_beam_width=ns.jasper_beam_width,
    )
    return config, bool(ns.dry_run)


def _validate_positive_options(ns: argparse.Namespace) -> None:
    for name in (
        "requests_per_user",
        "max_output_tokens",
        "top_k",
        "kv_max_model_len",
        "kv_max_position",
        "kv_block_size",
        "jasper_n_neighbors",
        "jasper_beam_width",
    ):
        if getattr(ns, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero.")
    if ns.warmup_batches < 0:
        raise ValueError("--warmup-batches must be at least zero.")
    if not 0 < ns.kv_gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between zero and one.")
    uses_jasper = any(condition_vector_backend(condition) == "jasper" for condition in ns.conditions)
    if uses_jasper and max(ns.jasper_beam_width, ns.top_k) > MAX_JASPER_BEAM_WIDTH:
        raise ValueError(
            f"Effective Jasper beam width must be at most {MAX_JASPER_BEAM_WIDTH}."
        )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._").lower()
    return normalized or "model"
