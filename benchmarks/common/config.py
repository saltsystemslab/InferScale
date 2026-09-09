"""JSON configuration shared by every benchmark: the runtime file and helpers.

``configs/runtime.json`` describes the machine (storage roots, build knobs,
model aliases, the judge server, environment variables). Each benchmark has
its own JSON files under ``configs/`` that reference models by alias and
embed an ``inferscale`` section for the library.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args, get_type_hints

from .paths import StorageConfig, StorageLayout, project_root, resolve_layout

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
JUDGE_LLM_API_KEY_ENV = "JUDGE_LLM_API_KEY"
EXTRACTION_LLM_API_KEY_ENV = "EXTRACTION_LLM_API_KEY"
DEFAULT_EXTRACTION_LLM_BASE_URL = "http://localhost:8000/v1"
REDACTED = "<redacted>"


class ConfigError(ValueError):
    """A configuration file is missing, malformed, or holds an invalid value."""


def load_json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON: {target} ({exc})") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must contain a JSON object: {target}")
    return raw


def require(section: Mapping[str, Any], key: str, expected: type | tuple[type, ...], where: str) -> Any:
    if key not in section:
        raise ConfigError(f"{where} is missing required key {key!r}.")
    return _typed(section[key], key, expected, where)


def optional(
    section: Mapping[str, Any],
    key: str,
    expected: type | tuple[type, ...],
    default: Any,
    where: str,
) -> Any:
    if key not in section or section[key] is None:
        return default
    return _typed(section[key], key, expected, where)


def reject_unknown_keys(section: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ConfigError(f"{where} has unknown keys: {', '.join(unknown)}.")


def section_of(data: Mapping[str, Any], key: str, where: str, *, required: bool = True) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        if required:
            raise ConfigError(f"{where} is missing required section {key!r}.")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}.{key} must be a JSON object.")
    return value


def expand_path(value: str | Path, *, root: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else root / expanded


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


def redact(data: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    for key in keys:
        if data.get(key):
            data[key] = REDACTED
    return data


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
    def from_json(cls, path: str | Path, *, root: Path | None = None) -> "RuntimeConfig":
        target = Path(path)
        return cls.from_dict(load_json_object(target), path=target, root=root)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        path: str | Path = "runtime.json",
        root: Path | None = None,
    ) -> "RuntimeConfig":
        where = "runtime"
        project = root if root is not None else project_root()
        reject_unknown_keys(
            data,
            ("storage", "build", "models", "reasoning_parsers", "judge_server", "environment"),
            where,
        )
        storage_section = section_of(data, "storage", where, required=False)
        reject_unknown_keys(
            storage_section,
            ("runtime_root", "cache_root", "results_root", "tmp_dir", "mem0_dir", "local_store_dir"),
            f"{where}.storage",
        )

        def storage_path(key: str) -> Path | None:
            value = optional(storage_section, key, str, None, f"{where}.storage")
            return expand_path(value, root=project) if value else None

        storage = StorageConfig(
            runtime_root=storage_path("runtime_root"),
            cache_root=storage_path("cache_root"),
            results_root=storage_path("results_root"),
            tmp_dir=storage_path("tmp_dir"),
            mem0_dir=storage_path("mem0_dir"),
            local_store_dir=storage_path("local_store_dir") or Path("/tmp"),
        )
        build_section = section_of(data, "build", where, required=False)
        reject_unknown_keys(
            build_section,
            ("cuda_module", "pytorch_index", "jasper_cuda_architectures", "constraints_file", "venv_dir"),
            f"{where}.build",
        )
        defaults = BuildConfig()
        build = BuildConfig(
            cuda_module=optional(build_section, "cuda_module", str, defaults.cuda_module, f"{where}.build"),
            pytorch_index=optional(build_section, "pytorch_index", str, defaults.pytorch_index, f"{where}.build"),
            jasper_cuda_architectures=optional(
                build_section, "jasper_cuda_architectures", str, defaults.jasper_cuda_architectures, f"{where}.build"
            ),
            constraints_file=optional(build_section, "constraints_file", str, defaults.constraints_file, f"{where}.build"),
            venv_dir=optional(build_section, "venv_dir", str, defaults.venv_dir, f"{where}.build"),
        )
        models = _string_map(section_of(data, "models", where, required=False), f"{where}.models")
        reasoning_parsers = _string_map(
            section_of(data, "reasoning_parsers", where, required=False), f"{where}.reasoning_parsers"
        )
        judge_section = section_of(data, "judge_server", where, required=False)
        reject_unknown_keys(
            judge_section,
            ("model", "base_url", "api_key", "tensor_parallel_size", "gpu_memory_utilization", "dtype", "max_model_len", "quantization"),
            f"{where}.judge_server",
        )
        judge_defaults = JudgeServerConfig()
        judge_server = JudgeServerConfig(
            model=optional(judge_section, "model", str, judge_defaults.model, f"{where}.judge_server"),
            base_url=optional(judge_section, "base_url", str, judge_defaults.base_url, f"{where}.judge_server"),
            api_key=optional(judge_section, "api_key", str, judge_defaults.api_key, f"{where}.judge_server"),
            tensor_parallel_size=optional(judge_section, "tensor_parallel_size", int, judge_defaults.tensor_parallel_size, f"{where}.judge_server"),
            gpu_memory_utilization=optional(judge_section, "gpu_memory_utilization", (int, float), judge_defaults.gpu_memory_utilization, f"{where}.judge_server"),
            dtype=optional(judge_section, "dtype", str, judge_defaults.dtype, f"{where}.judge_server"),
            max_model_len=optional(judge_section, "max_model_len", int, judge_defaults.max_model_len, f"{where}.judge_server"),
            quantization=optional(judge_section, "quantization", str, None, f"{where}.judge_server"),
        )
        environment = _string_map(section_of(data, "environment", where, required=False), f"{where}.environment")
        return cls(
            storage=storage,
            layout=resolve_layout(storage, root=project),
            build=build,
            models=models,
            reasoning_parsers=reasoning_parsers,
            judge_server=judge_server,
            environment=environment,
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
        """Short run-id label: the alias when one was given, else a slug of the id's last segment."""
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


def default_runtime_config_path() -> Path:
    return project_root() / "configs" / "runtime.json"


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    return RuntimeConfig.from_json(Path(path) if path is not None else default_runtime_config_path())


@dataclass(slots=True)
class JudgeConfig:
    provider: str = "vllm"
    model: str = "google/gemma-2-9b-it"
    base_url: str | None = "http://127.0.0.1:8000/v1"
    api_key: str | None = "token-abc123"
    max_tokens: int = 4
    with_evidence: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": REDACTED if self.api_key else None,
            "max_tokens": self.max_tokens,
            "with_evidence": self.with_evidence,
        }


def judge_config(
    section: Mapping[str, Any],
    runtime: RuntimeConfig,
    *,
    where: str,
) -> JudgeConfig:
    reject_unknown_keys(section, ("provider", "model", "base_url", "max_tokens", "with_evidence"), where)
    provider = optional(section, "provider", str, "vllm", where)
    if provider not in {"vllm", "none"}:
        raise ConfigError(f"{where}.provider must be vllm or none, got {provider!r}.")
    max_tokens = optional(section, "max_tokens", int, 4, where)
    if max_tokens < 1:
        raise ConfigError(f"{where}.max_tokens must be >= 1.")
    configured_base_url = optional(section, "base_url", str, runtime.judge_server.base_url, where)
    return JudgeConfig(
        provider=provider,
        model=optional(section, "model", str, runtime.judge_server.model, where),
        base_url=configured_base_url,
        api_key=runtime.judge_api_key(),
        max_tokens=max_tokens,
        with_evidence=optional(section, "with_evidence", bool, False, where),
    )


def _typed(value: Any, key: str, expected: type | tuple[type, ...], where: str) -> Any:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if bool not in expected_types and isinstance(value, bool):
        raise ConfigError(f"{where}.{key} must be {_describe(expected_types)}, got a boolean.")
    if not isinstance(value, expected_types):
        raise ConfigError(
            f"{where}.{key} must be {_describe(expected_types)}, got {type(value).__name__}."
        )
    return value


def _describe(expected_types: tuple[type, ...]) -> str:
    names = {int: "an integer", float: "a number", str: "a string", bool: "a boolean", list: "a list", dict: "an object"}
    return " or ".join(names.get(item, item.__name__) for item in expected_types)


def _string_map(section: Mapping[str, Any], where: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in section.items():
        if not isinstance(value, str):
            raise ConfigError(f"{where}.{key} must be a string.")
        result[str(key)] = value
    return result


def int_list(section: Mapping[str, Any], key: str, where: str, *, default: Sequence[int] | None = None) -> tuple[int, ...]:
    if key not in section or section[key] is None:
        if default is None:
            raise ConfigError(f"{where} is missing required key {key!r}.")
        return tuple(default)
    values = section[key]
    if not isinstance(values, list) or not values or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        raise ConfigError(f"{where}.{key} must be a non-empty list of integers.")
    if len(set(values)) != len(values):
        raise ConfigError(f"{where}.{key} must not contain duplicates.")
    return tuple(values)


def str_list(section: Mapping[str, Any], key: str, where: str, *, default: Sequence[str] | None = None) -> tuple[str, ...]:
    if key not in section or section[key] is None:
        if default is None:
            raise ConfigError(f"{where} is missing required key {key!r}.")
        return tuple(default)
    values = section[key]
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        raise ConfigError(f"{where}.{key} must be a non-empty list of strings.")
    if len(set(values)) != len(values):
        raise ConfigError(f"{where}.{key} must not contain duplicates.")
    return tuple(values)


def inferscale_section(
    data: Mapping[str, Any],
    *,
    model: str,
    top_k: int,
    context_window: int,
    where: str,
) -> Any:
    """Build the library config with model, retrieval depth, and context window from the run."""
    from inferscale.v1 import InferScaleConfig

    section = dict(section_of(data, "inferscale", where))
    for key in ("model", "top_k", "context_window"):
        if key in section:
            raise ConfigError(f"{where}.inferscale.{key} is set from the top level; remove it from the section.")
    section["model"] = model
    section["top_k"] = top_k
    section["context_window"] = context_window
    try:
        _validate_dataclass_types(section, InferScaleConfig, f"{where}.inferscale")
        return InferScaleConfig.from_dict(section)
    except ValueError as exc:
        raise ConfigError(f"{where}.inferscale: {exc}") from exc



def _validate_dataclass_types(data: Mapping[str, Any], schema: type, where: str) -> None:
    """Reject JSON type mismatches before dataclass constructors use values."""
    hints = get_type_hints(schema)
    reject_unknown_keys(data, tuple(hints), where)
    for name, value in data.items():
        expected = hints[name]
        if is_dataclass(expected):
            if not isinstance(value, dict):
                raise ConfigError(f"{where}.{name} must be a JSON object.")
            _validate_dataclass_types(value, expected, f"{where}.{name}")
            continue
        choices = get_args(expected) or (expected,)
        if float in choices:
            choices = (*choices, int)
        _typed(value, name, choices, where)


def jsonable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    return value


__all__ = [
    "BuildConfig",
    "ConfigError",
    "DEFAULT_EXTRACTION_LLM_BASE_URL",
    "EXTRACTION_LLM_API_KEY_ENV",
    "JUDGE_LLM_API_KEY_ENV",
    "JudgeConfig",
    "JudgeServerConfig",
    "OPENAI_API_KEY_ENV",
    "REDACTED",
    "RuntimeConfig",
    "default_runtime_config_path",
    "embedding_api_key",
    "expand_path",
    "extraction_llm_api_key",
    "inferscale_section",
    "int_list",
    "judge_config",
    "judge_llm_api_key",
    "jsonable_value",
    "load_json_object",
    "load_runtime_config",
    "optional",
    "redact",
    "reject_unknown_keys",
    "require",
    "section_of",
    "slug",
    "stamp_now",
    "str_list",
]
