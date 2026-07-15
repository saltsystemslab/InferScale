from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cache_root(root: str | Path | None = None) -> Path:
    if os.environ.get("BENCHMARK_CACHE_ROOT"):
        return Path(os.environ["BENCHMARK_CACHE_ROOT"])
    if os.environ.get("BENCHMARK_RUNTIME_ROOT"):
        return Path(os.environ["BENCHMARK_RUNTIME_ROOT"]) / ".cache"
    if os.environ.get("SCRATCH_ROOT"):
        return Path(os.environ["SCRATCH_ROOT"]) / "cache"
    if _is_workspace_project(root):
        return Path("/workspace/.cache")
    return project_root() / ".cache"


def default_results_root(root: str | Path | None = None) -> Path:
    if os.environ.get("BENCHMARK_RESULTS_ROOT"):
        return Path(os.environ["BENCHMARK_RESULTS_ROOT"])
    if os.environ.get("BENCHMARK_RUNTIME_ROOT"):
        return Path(os.environ["BENCHMARK_RUNTIME_ROOT"]) / "results"
    if os.environ.get("SCRATCH_ROOT"):
        return Path(os.environ["SCRATCH_ROOT"]) / "results"
    if _is_workspace_project(root):
        return Path("/workspace/results")
    return Path("results")


def default_tmp_dir(root: str | Path | None = None) -> Path:
    if os.environ.get("TMPDIR"):
        return Path(os.environ["TMPDIR"])
    if os.environ.get("BENCHMARK_RUNTIME_ROOT"):
        return Path(os.environ["BENCHMARK_RUNTIME_ROOT"]) / "tmp"
    if os.environ.get("SCRATCH_ROOT"):
        return Path(os.environ["SCRATCH_ROOT"]) / "tmp"
    if _is_workspace_project(root):
        return Path("/workspace/tmp")
    return project_root() / "tmp"


def default_mem0_dir(root: str | Path | None = None) -> Path:
    if os.environ.get("MEM0_DIR"):
        return Path(os.environ["MEM0_DIR"])
    return default_cache_root(root) / "mem0"


def local_store_scratch_dir(run_id: str) -> Path:
    """Pod-local scratch root for rebuild-per-run vector stores.

    Deliberately NOT TMPDIR-based: remote scratch env points TMPDIR at the
    shared network volume, whose FUSE mount cannot support sqlite file
    locking. /tmp is container-local on the pods.
    """
    root = os.environ.get("LOCOMO_LOCAL_STORE_DIR") or "/tmp"
    return Path(root) / "locomo-jasper-stores" / run_id


def default_embedding_cache_dir(root: str | Path | None = None) -> Path:
    return default_cache_root(root) / "embeddings"


def default_memory_llm_cache_dir(root: str | Path | None = None) -> Path:
    return default_cache_root(root) / "mem0-inference"


def default_mem0_dir_string() -> str:
    return str(default_mem0_dir())


def configure_runtime_environment(root: str | Path | None = None) -> None:
    cache_root = default_cache_root(root)
    os.environ.setdefault("BENCHMARK_CACHE_ROOT", str(cache_root))
    os.environ.setdefault("BENCHMARK_RESULTS_ROOT", str(default_results_root(root)))
    os.environ.setdefault("MEM0_DIR", str(default_mem0_dir(root)))
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    os.environ.setdefault("TMPDIR", str(default_tmp_dir(root)))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(os.environ["HF_HOME"]) / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache_root / "cuda"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(cache_root / "vllm"))
    os.environ.setdefault("VLLM_CONFIG_ROOT", str(cache_root / "vllm_config"))


def _is_workspace_project(root: str | Path | None = None) -> bool:
    candidate = Path(root) if root is not None else project_root()
    try:
        candidate = candidate.resolve()
    except OSError:
        candidate = candidate.absolute()
    workspace = Path("/workspace")
    return candidate == workspace or workspace in candidate.parents
