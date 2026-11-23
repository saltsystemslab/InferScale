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

WHERE = "runtime"
RUNTIME_SECTIONS = (
    "storage",
    "build",
    "models",
    "reasoning_parsers",
    "judge_server",
    "environment",
)
STORAGE_WHERE = f"{WHERE}.storage"
STORAGE_KEYS = (
    "runtime_root",
    "cache_root",
    "results_root",
    "tmp_dir",
    "mem0_dir",
    "local_store_dir",
)


@dataclass(slots=True, frozen=True)
class BuildConfig:
    cuda_module: str = ""
    pytorch_index: str = "https://download.pytorch.org/whl/cu128"
    jasper_cuda_architectures: str = "native"
    constraints_file: str = "constraints-cu128.txt"
    venv_dir: str = ".venv"


@dataclass(slots=True, frozen=True)
class JudgeServerConfig:
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
        runtime_path = Path(path)
        return cls.from_dict(load_json_object(runtime_path), path=runtime_path, root=root)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        path: str | Path = "runtime.json",
        root: Path | None = None,
    ) -> RuntimeConfig:
        project = root if root is not None else project_root()
        reject_unknown_keys(data, RUNTIME_SECTIONS, WHERE)
        storage = _storage_config(_optional_section(data, "storage"), project)
        return cls(
            storage=storage,
            layout=resolve_layout(storage, root=project),
            build=_dataclass_section(data, "build", BuildConfig),
            models=_string_section(data, "models"),
            reasoning_parsers=_string_section(data, "reasoning_parsers"),
            judge_server=_dataclass_section(data, "judge_server", JudgeServerConfig),
            environment=_string_section(data, "environment"),
            path=Path(path),
            root=project,
        )

    def resolve_model(self, name: str) -> str:
        stripped = name.strip()
        if not stripped:
            raise ConfigError("model must be a non-empty alias, Hugging Face id, or local path.")
        return self.models.get(stripped, stripped)

    def model_label(self, name: str) -> str:
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
        for name, value in self.environment.items():
            os.environ[name] = value
        self.layout.apply_environment()


def _optional_section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    return section_of(data, name, WHERE, required=False)


def _dataclass_section(data: Mapping[str, Any], name: str, schema: type) -> Any:
    return parse_dataclass_section(_optional_section(data, name), schema, f"{WHERE}.{name}")


def _string_section(data: Mapping[str, Any], name: str) -> dict[str, str]:
    return string_map(_optional_section(data, name), f"{WHERE}.{name}")


def _storage_config(section: Mapping[str, Any], project: Path) -> StorageConfig:
    reject_unknown_keys(section, STORAGE_KEYS, STORAGE_WHERE)

    def storage_path(key: str) -> Path | None:
        authored_path = optional(section, key, str, None, STORAGE_WHERE)
        return expand_path(authored_path, root=project) if authored_path else None

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
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._").lower()
    return normalized or "model"


def embedding_api_key() -> str | None:
    return os.environ.get(OPENAI_API_KEY_ENV) or None


def judge_llm_api_key() -> str | None:
    return os.environ.get(JUDGE_LLM_API_KEY_ENV) or None


def extraction_llm_api_key() -> str | None:
    return os.environ.get(EXTRACTION_LLM_API_KEY_ENV) or None
