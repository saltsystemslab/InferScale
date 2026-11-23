from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOCAL_STORE_DIR = Path("/tmp")
RUN_CONFIG_FILE = "config.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True, frozen=True)
class StorageConfig:
    runtime_root: Path | None = None
    cache_root: Path | None = None
    results_root: Path | None = None
    tmp_dir: Path | None = None
    mem0_dir: Path | None = None
    local_store_dir: Path = DEFAULT_LOCAL_STORE_DIR


@dataclass(slots=True, frozen=True)
class StorageLayout:
    runtime_root: Path
    cache_root: Path
    results_root: Path
    tmp_dir: Path
    mem0_dir: Path
    local_store_dir: Path

    @property
    def embedding_cache_dir(self) -> Path:
        return self.cache_root / "embeddings"

    @property
    def memory_llm_cache_dir(self) -> Path:
        return self.cache_root / "mem0-inference"

    @property
    def rag_kv_chunk_cache_root(self) -> Path:
        return self.cache_root / "rag-kv-chunks"

    def local_store_scratch_dir(self, run_id: str) -> Path:
        return self.local_store_dir / "inferscale-bench-stores" / run_id

    def environment(self) -> dict[str, str]:
        hf_home = self.cache_root / "huggingface"
        return {
            "MEM0_DIR": str(self.mem0_dir),
            "TMPDIR": str(self.tmp_dir),
            "PIP_CACHE_DIR": str(self.cache_root / "pip"),
            "XDG_CACHE_HOME": str(self.cache_root / "xdg"),
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
            "TORCH_HOME": str(self.cache_root / "torch"),
            "TRITON_CACHE_DIR": str(self.cache_root / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(self.cache_root / "torchinductor"),
            "TORCH_EXTENSIONS_DIR": str(self.cache_root / "torch_extensions"),
            "CUDA_CACHE_PATH": str(self.cache_root / "cuda"),
            "VLLM_CACHE_ROOT": str(self.cache_root / "vllm"),
            "VLLM_CONFIG_ROOT": str(self.cache_root / "vllm_config"),
            "FLASHINFER_WORKSPACE_BASE": str(self.runtime_root),
            "FLASHINFER_WORKSPACE_DIR": str(self.cache_root / "flashinfer"),
        }

    def apply_environment(self) -> None:
        for name, value in self.environment().items():
            os.environ[name] = value

    def create_directories(self) -> None:
        directories = [self.cache_root, self.results_root, self.tmp_dir, self.mem0_dir]
        directories.extend(Path(location) for location in self.environment().values())
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def resolve_layout(storage: StorageConfig, *, root: Path | None = None) -> StorageLayout:
    project = root if root is not None else project_root()
    runtime_root = storage.runtime_root if storage.runtime_root is not None else project
    cache_root = storage.cache_root if storage.cache_root is not None else runtime_root / ".cache"
    results_root = (
        storage.results_root if storage.results_root is not None else runtime_root / "results"
    )
    tmp_dir = storage.tmp_dir if storage.tmp_dir is not None else runtime_root / "tmp"
    mem0_dir = storage.mem0_dir if storage.mem0_dir is not None else cache_root / "mem0"
    return StorageLayout(
        runtime_root=runtime_root,
        cache_root=cache_root,
        results_root=results_root,
        tmp_dir=tmp_dir,
        mem0_dir=mem0_dir,
        local_store_dir=storage.local_store_dir,
    )


def mem0_dir_from_environment() -> Path:
    configured = os.environ.get("MEM0_DIR")
    if configured:
        return Path(configured)
    return project_root() / ".cache" / "mem0"
