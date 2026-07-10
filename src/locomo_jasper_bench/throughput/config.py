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
from ..runtime_paths import default_results_root

ALL_CONDITIONS = (
    "no_memory",
    "prompt_injection",
    "kv_injection",
    "mem0",
)


@dataclass(frozen=True, order=True, slots=True)
class BenchmarkPoint:
    num_users: int
    memory_tokens: int

    def __post_init__(self) -> None:
        if self.num_users <= 0:
            raise ValueError("num_users must be greater than zero.")
        if self.memory_tokens <= 0:
            raise ValueError("memory_tokens must be greater than zero.")

    def __str__(self) -> str:
        return f"{self.num_users}:{self.memory_tokens}"


DEFAULT_MATRIX = (
    BenchmarkPoint(10, 512),
    BenchmarkPoint(10, 1024),
    BenchmarkPoint(10, 2048),
    BenchmarkPoint(10, 4096),
    BenchmarkPoint(25, 512),
    BenchmarkPoint(25, 1024),
    BenchmarkPoint(25, 2048),
    BenchmarkPoint(25, 4096),
    BenchmarkPoint(50, 512),
    BenchmarkPoint(50, 1024),
    BenchmarkPoint(50, 2048),
    BenchmarkPoint(50, 4096),
    BenchmarkPoint(100, 512),
    BenchmarkPoint(100, 1024),
    BenchmarkPoint(100, 2048),
)
DEFAULT_MATRIX_TEXT = ",".join(str(point) for point in DEFAULT_MATRIX)


@dataclass(slots=True)
class ThroughputConfig:
    model: str
    model_label: str
    results_dir: Path
    run_id: str
    conditions: tuple[str, ...] = ALL_CONDITIONS
    matrix: tuple[BenchmarkPoint, ...] = DEFAULT_MATRIX
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
    vector_backend: str = "qdrant"
    vector_distance: str = "ip"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    jasper_n_neighbors: int = 64
    jasper_alpha: float = 1.0
    jasper_workspace_budget: str = "10GB"
    jasper_beam_width: int = 64

    @property
    def run_dir(self) -> Path:
        return self.results_dir / "throughput" / self.run_id

    def to_jsonable(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["results_dir"] = str(self.results_dir)
        data["matrix"] = [asdict(point) for point in self.matrix]
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
        data["results_dir"] = Path(data["results_dir"])
        data["conditions"] = tuple(data["conditions"])
        data["matrix"] = tuple(BenchmarkPoint(**point) for point in data["matrix"])
        if data.get("embedding_api_key") == "<redacted>":
            data["embedding_api_key"] = (
                os.environ.get("LOCOMO_THROUGHPUT_EMBEDDING_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
        return cls(**data)


def parse_matrix(value: str | Iterable[str]) -> tuple[BenchmarkPoint, ...]:
    raw_parts: list[str] = []
    values = [value] if isinstance(value, str) else list(value)
    for item in values:
        raw_parts.extend(part.strip() for part in re.split(r"[;,\s]+", item) if part.strip())
    if not raw_parts:
        raise ValueError("The benchmark matrix cannot be empty.")

    points: list[BenchmarkPoint] = []
    seen: set[BenchmarkPoint] = set()
    for raw in raw_parts:
        separator = ":" if ":" in raw else "x" if "x" in raw.lower() else None
        if separator is None:
            raise ValueError(f"Invalid matrix point {raw!r}; expected USERS:TOKENS.")
        if separator == "x":
            parts = re.split(r"[xX]", raw)
        else:
            parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid matrix point {raw!r}; expected USERS:TOKENS.")
        try:
            point = BenchmarkPoint(int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise ValueError(f"Invalid matrix point {raw!r}: {exc}") from exc
        if point in seen:
            raise ValueError(f"Duplicate matrix point: {point}")
        points.append(point)
        seen.add(point)
    return tuple(points)


def matrix_text(points: Iterable[BenchmarkPoint]) -> str:
    return ",".join(str(point) for point in points)


def default_run_id(model_label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_slug(model_label)}-{stamp}"


def parse_args(argv: list[str] | None = None) -> tuple[ThroughputConfig, bool]:
    parser = argparse.ArgumentParser(
        prog="locomo-throughput-bench",
        description="Run synthetic multi-user throughput benchmarks for vLLM memory conditions.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LOCOMO_VLLM_MODEL", "llama"),
        help="Hugging Face model id, local path, or configured alias: llama, mistral, qwen, qwen3-14b.",
    )
    parser.add_argument("--results-dir", type=Path, default=default_results_root())
    parser.add_argument("--run-id")
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS, default=list(ALL_CONDITIONS))
    parser.add_argument(
        "--matrix",
        default=os.environ.get("THROUGHPUT_MATRIX", DEFAULT_MATRIX_TEXT),
        help="Comma-separated USERS:TOKENS points.",
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
    parser.add_argument("--vector-backend", choices=("jasper", "qdrant"), default="qdrant")
    parser.add_argument("--vector-distance", choices=("ip", "l2"), default="ip")
    parser.add_argument("--embedding-model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--jasper-n-neighbors", type=int, default=64)
    parser.add_argument("--jasper-alpha", type=float, default=1.0)
    parser.add_argument("--jasper-workspace-budget", default="10GB")
    parser.add_argument("--jasper-beam-width", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")

    ns = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        matrix = parse_matrix(ns.matrix)
        _validate_positive_options(ns)
        _validate_matrix(matrix, ns)
    except ValueError as exc:
        parser.error(str(exc))

    raw_model = ns.model.strip()
    model_label = ANSWER_MODEL_NAME_ALIASES.get(raw_model.lower(), raw_model.rsplit("/", 1)[-1])
    config = ThroughputConfig(
        model=resolve_answer_model(raw_model),
        model_label=model_label,
        results_dir=ns.results_dir,
        run_id=ns.run_id or default_run_id(model_label),
        conditions=tuple(dict.fromkeys(ns.conditions)),
        matrix=matrix,
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
        vector_backend=ns.vector_backend,
        vector_distance=ns.vector_distance,
        embedding_model=ns.embedding_model,
        embedding_api_key=ns.embedding_api_key,
        embedding_base_url=ns.embedding_base_url,
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
    if ns.vector_backend == "jasper" and max(ns.jasper_beam_width, ns.top_k) > MAX_JASPER_BEAM_WIDTH:
        raise ValueError(
            f"Effective Jasper beam width must be at most {MAX_JASPER_BEAM_WIDTH}."
        )


def _validate_matrix(matrix: tuple[BenchmarkPoint, ...], ns: argparse.Namespace) -> None:
    largest_memory = max(point.memory_tokens for point in matrix)
    if largest_memory >= ns.kv_max_model_len:
        raise ValueError("Every memory token count must be smaller than --max-model-len.")
    if largest_memory > ns.kv_max_position:
        raise ValueError("Every memory token count must be at most --kv-max-position.")
    if "kv_injection" in ns.conditions:
        unaligned = [point for point in matrix if point.memory_tokens % ns.kv_block_size]
        if unaligned:
            values = ", ".join(str(point) for point in unaligned[:5])
            raise ValueError(
                f"KV memory token counts must be divisible by --kv-block-size={ns.kv_block_size}: {values}"
            )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._").lower()
    return normalized or "model"
