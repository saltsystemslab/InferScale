from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError, expand_path, load_json_object, optional
from benchmarks.common.paths import project_root

ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def runtime_path_for_config(config_path: Path) -> Path | None:
    config = load_json_object(config_path)
    own_runtime_path = _configured_path(config, "runtime_config", str(config_path))
    if own_runtime_path is not None:
        return own_runtime_path
    manifest_path = _configured_path(config, "manifest", str(config_path))
    if manifest_path is None:
        return None
    saved_manifest = load_json_object(manifest_path)
    return _configured_path(saved_manifest, "runtime_config", str(manifest_path))


def _configured_path(config: Mapping[str, Any], key: str, where: str) -> Path | None:
    configured = optional(config, key, str, None, where)
    if configured is None:
        return None
    if not configured.strip():
        raise ConfigError(f"{where}: {key} must be a non-empty path.")
    return expand_path(configured, root=project_root())


def runtime_shell_exports(runtime: RuntimeConfig, *, prepare: bool = False) -> str:
    exports = {
        **runtime.environment,
        **runtime.layout.environment(),
        **_runtime_and_build_exports(runtime),
    }
    for name in exports:
        if not ENVIRONMENT_NAME.fullmatch(name):
            raise ConfigError(f"Invalid runtime environment name: {name!r}")
    if prepare:
        runtime.layout.create_directories()
    return "\n".join(f"export {name}={shlex.quote(str(value))}" for name, value in exports.items())


def _runtime_and_build_exports(runtime: RuntimeConfig) -> dict[str, str]:
    return {
        "INFERSCALE_RUNTIME_CONFIG": str(runtime.path.resolve()),
        "BENCHMARK_RUNTIME_ROOT": str(runtime.layout.runtime_root),
        "BENCHMARK_CACHE_ROOT": str(runtime.layout.cache_root),
        "BENCHMARK_RESULTS_ROOT": str(runtime.layout.results_root),
        "CUDA_MODULE": runtime.build.cuda_module,
        "PYTORCH_INDEX": runtime.build.pytorch_index,
        "JASPER_CUDA_ARCHITECTURES": runtime.build.jasper_cuda_architectures,
        "CONSTRAINTS_FILE": str(runtime.root / runtime.build.constraints_file),
        "VENV_DIR": str(runtime.root / runtime.build.venv_dir),
    }
