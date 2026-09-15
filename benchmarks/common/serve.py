"""Start the judge server described by the runtime JSON configuration."""
from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

from benchmarks.common.config import ConfigError, load_json_object, load_runtime_config, optional, reject_unknown_keys
from benchmarks.common.environment import runtime_path_for_config
from urllib.parse import urlsplit


def serve(config_path: Path) -> int:
    """Start or preview the server described by a JSON workflow file."""
    data = load_json_object(config_path)
    reject_unknown_keys(data, ("runtime_config", "dry_run"), "serve")
    dry_run = optional(data, "dry_run", bool, False, "serve")
    runtime = load_runtime_config(runtime_path_for_config(config_path))
    runtime.apply_environment()
    config = runtime.judge_server
    endpoint = urlsplit(config.base_url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ConfigError("Local judge serving requires judge_server.base_url to use a local HTTP endpoint.")
    command = ["vllm", "serve", runtime.resolve_model(config.model), "--host", endpoint.hostname,
               "--port", str(endpoint.port or 80), "--trust-remote-code", "--dtype", config.dtype,
               "--max-model-len", str(config.max_model_len), "--tensor-parallel-size", str(config.tensor_parallel_size),
               "--gpu-memory-utilization", str(config.gpu_memory_utilization)]
    if config.quantization:
        command += ["--quantization", config.quantization]
    print(shlex.join(command), flush=True)
    if dry_run:
        return 0
    runtime.layout.create_directories()
    if runtime.judge_api_key():
        command += ["--api-key", runtime.judge_api_key()]
    return subprocess.call(command)

