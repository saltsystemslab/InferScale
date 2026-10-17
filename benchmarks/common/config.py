"""The runtime JSON file and the credentials that accompany it.

``configs/runtime.json`` describes the machine: storage roots, build settings,
model aliases, the judge server, and environment variables to export. Each
benchmark has its own JSON files under ``configs/`` that reference models by
alias. Credentials are the one thing read from the environment (``.env``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.common.json_fields import (
    ConfigError,
    expand_path,
    load_json_object,
    optional,
    parse_dataclass_section,
    reject_unknown_keys,
    section_of,
    string_map,
)
from benchmarks.common.paths import (
    DEFAULT_LOCAL_STORE_DIR,
    StorageConfig,
    StorageLayout,
    project_root,
    resolve_layout,
)

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
JUDGE_LLM_API_KEY_ENV = "JUDGE_LLM_API_KEY"
EXTRACTION_LLM_API_KEY_ENV = "EXTRACTION_LLM_API_KEY"
DEFAULT_EXTRACTION_LLM_BASE_URL = "http://localhost:8000/v1"
REDACTED = "<redacted>"

RUNTIME_SECTIONS = (
    "storage",
    "build",
    "models",
    "reasoning_parsers",
    "judge_server",
    "environment",
)


@dataclass(slots=True, frozen=True)
class BuildConfig:
    """Installation settings consumed by scripts/setup_remote.sh."""

    cuda_module: str = ""
    pytorch_index: str = "https://download.pytorch.org/whl/cu128"
    jasper_cuda_architectures: str = "native"
    constraints_file: str = "constraints-cu128.txt"
    venv_dir: str = ".venv"


@dataclass(slots=True, frozen=True)
class JudgeServerConfig:
    """The local vLLM judge server started by scripts/serve_vllm.sh."""

    model: str = "google/gemma-2-9b-it"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "token-abc123"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    dtype: str = "auto"
    max_model_len: int = 8192
    quantization: str | None = None


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    storage: StorageConfig
    layout: StorageLayout
    build: BuildConfig
    models: dict[str, str]
    reasoning_parsers: dict[str, str]
    judge_server: JudgeServerConfig
    environment: dict[str, str]
    path: Path
    root: Path

    @classmethod
    def from_json(cls, path: str | Path, *, root: Path | None = None) -> RuntimeConfig:
        target = Path(path)
        return cls.from_dict(load_json_object(target), path=target, root=root)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        path: str | Path = "runtime.json",
        root: Path | None = None,
    ) -> RuntimeConfig:
        where = "runtime"
        project = root if root is not None else project_root()
        reject_unknown_keys(data, RUNTIME_SECTIONS, where)
        storage = _storage_config(section_of(data, "storage", where, required=False), project)
        return cls(
            storage=storage,
            layout=resolve_layout(storage, root=project),
            build=parse_dataclass_section(
                section_of(data, "build", where, required=False), BuildConfig, f"{where}.build"
            ),
            models=string_map(section_of(data, "models", where, required=False), f"{where}.models"),
            reasoning_parsers=string_map(
                section_of(data, "reasoning_parsers", where, required=False),
                f"{where}.reasoning_parsers",
            ),
            judge_server=parse_dataclass_section(
                section_of(data, "judge_server", where, required=False),
                JudgeServerConfig,
                f"{where}.judge_server",
            ),
            environment=string_map(
                section_of(data, "environment", where, required=False), f"{where}.environment"
            ),
            path=Path(path),
            root=project,
        )

    def resolve_model(self, name: str) -> str:
        """Alias table lookup; Hugging Face ids and local paths pass through."""
        stripped = name.strip()
        if not stripped:
            raise ConfigError("model must be a non-empty alias, Hugging Face id, or local path.")
        return self.models.get(stripped, stripped)

    def model_label(self, name: str) -> str:
        """Short run-id label: the alias when one exists, else a slug of the id's last segment."""
        stripped = name.strip()
        if stripped in self.models:
            return stripped
        for alias, model_id in self.models.items():
            if model_id == stripped:
                return alias
        return slug(stripped.rsplit("/", 1)[-1])

    def reasoning_parser(self, name: str) -> str | None:
        return self.reasoning_parsers.get(self.model_label(name))

    def judge_api_key(self) -> str:
        return judge_llm_api_key() or self.judge_server.api_key

    def apply_environment(self) -> None:
        """Export cache locations and the environment block before heavy imports."""
        for key, value in self.environment.items():
            os.environ[key] = value
        self.layout.apply_environment()


def _storage_config(section: Mapping[str, Any], project: Path) -> StorageConfig:
    where = "runtime.storage"
    reject_unknown_keys(
        section,
        ("runtime_root", "cache_root", "results_root", "tmp_dir", "mem0_dir", "local_store_dir"),
        where,
    )

    def storage_path(key: str) -> Path | None:
        value = optional(section, key, str, None, where)
        return expand_path(value, root=project) if value else None

    return StorageConfig(
        runtime_root=storage_path("runtime_root"),
        cache_root=storage_path("cache_root"),
        results_root=storage_path("results_root"),
        tmp_dir=storage_path("tmp_dir"),
        mem0_dir=storage_path("mem0_dir"),
        local_store_dir=storage_path("local_store_dir") or DEFAULT_LOCAL_STORE_DIR,
    )


def default_runtime_config_path() -> Path:
    return project_root() / "configs" / "runtime.json"


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    return RuntimeConfig.from_json(
        Path(path) if path is not None else default_runtime_config_path()
    )


def stamp_now() -> str:
    """UTC timestamp used to make run ids and launch folders unique."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug(value: str) -> str:
    """Lowercase the value and replace every run of other characters with a dash."""
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._").lower()
    return normalized or "model"


def embedding_api_key() -> str | None:
    return os.environ.get(OPENAI_API_KEY_ENV) or None


def judge_llm_api_key() -> str | None:
    return os.environ.get(JUDGE_LLM_API_KEY_ENV) or None


def extraction_llm_api_key() -> str | None:
    return os.environ.get(EXTRACTION_LLM_API_KEY_ENV) or None
