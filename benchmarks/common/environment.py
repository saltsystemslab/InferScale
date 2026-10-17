"""Shell exports derived from the runtime JSON, used by scripts/environment.sh.

The scripts source ``.env`` for credentials and then evaluate these exports, so
paths and build settings always come from ``configs/runtime.json``.
"""

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
    """The runtime JSON a workflow file selects: its own, else its saved manifest's, else None."""
    source = load_json_object(config_path)
    runtime = _configured_path(source, "runtime_config", str(config_path))
    if runtime is not None:
        return runtime
    manifest = _configured_path(source, "manifest", str(config_path))
    if manifest is None:
        return None
    return _configured_path(load_json_object(manifest), "runtime_config", str(manifest))


def _configured_path(source: Mapping[str, Any], key: str, where: str) -> Path | None:
    value = optional(source, key, str, None, where)
    if value is None:
        return None
    if not value.strip():
        raise ConfigError(f"{where}: {key} must be a non-empty path.")
    return expand_path(value, root=project_root())


def runtime_shell_exports(runtime: RuntimeConfig, *, prepare: bool = False) -> str:
    """Quoted ``export`` lines for every value the shell scripts take from the runtime JSON."""
    if prepare:
        runtime.layout.create_directories()
    values = dict(runtime.environment)
    values.update(runtime.layout.environment())
    values.update(
        {
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
    )
    for name in values:
        if not ENVIRONMENT_NAME.fullmatch(name):
            raise ConfigError(f"Invalid runtime environment name: {name!r}")
    return "\n".join(f"export {name}={shlex.quote(str(value))}" for name, value in values.items())
