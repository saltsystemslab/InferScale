"""Programmatic runtime selection and shell exports for fixed JSON workflows."""
from __future__ import annotations

from pathlib import Path
import re
import shlex

from benchmarks.common.config import ConfigError, RuntimeConfig, expand_path, load_json_object, optional
from benchmarks.common.paths import project_root


def runtime_path_for_config(config_path: Path) -> Path | None:
    """Select an explicit plan runtime, then its saved manifest runtime."""
    source = load_json_object(config_path)
    runtime = optional(source, "runtime_config", str, None, str(config_path))
    if runtime is not None and not runtime.strip():
        raise ConfigError(f"{config_path}: runtime_config must be a non-empty path.")
    if runtime:
        return expand_path(runtime, root=project_root())
    manifest = optional(source, "manifest", str, None, str(config_path))
    if manifest is not None and not manifest.strip():
        raise ConfigError(f"{config_path}: manifest must be a non-empty path.")
    if manifest:
        manifest_path = expand_path(manifest, root=project_root())
        saved = load_json_object(manifest_path)
        runtime = optional(saved, "runtime_config", str, None, str(manifest_path))
        if runtime is not None and not runtime.strip():
            raise ConfigError(f"{manifest_path}: runtime_config must be a non-empty path.")
        if runtime:
            return expand_path(runtime, root=project_root())
    return None


def shell_exports(runtime: RuntimeConfig, *, prepare: bool = False) -> str:
    """Return safely quoted exports derived entirely from the runtime JSON."""
    if prepare:
        runtime.layout.create_directories()
    values = dict(runtime.environment)
    values.update(runtime.layout.environment())
    values.update({
        "INFERSCALE_RUNTIME_CONFIG": str(runtime.path.resolve()),
        "BENCHMARK_RUNTIME_ROOT": str(runtime.layout.runtime_root),
        "BENCHMARK_CACHE_ROOT": str(runtime.layout.cache_root),
        "BENCHMARK_RESULTS_ROOT": str(runtime.layout.results_root),
        "CUDA_MODULE": runtime.build.cuda_module,
        "PYTORCH_INDEX": runtime.build.pytorch_index,
        "JASPER_CUDA_ARCHITECTURES": runtime.build.jasper_cuda_architectures,
        "CONSTRAINTS_FILE": str(runtime.root / runtime.build.constraints_file),
        "VENV_DIR": str(runtime.root / runtime.build.venv_dir),
    })
    exports = []
    for name, value in values.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ConfigError(f"Invalid runtime environment name: {name!r}")
        exports.append(f"export {name}={shlex.quote(str(value))}")
    return "\n".join(exports)
