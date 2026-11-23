from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from loguru import logger

from benchmarks.common.config import RuntimeConfig, load_runtime_config
from benchmarks.common.environment import runtime_path_for_config
from benchmarks.common.json_fields import load_json_object, optional, reject_unknown_keys
from benchmarks.common.logging import configure_logging
from benchmarks.common.vllm_server import require_local_http_endpoint, vllm_serve_command

WHERE = "serve"
SERVE_KEYS = ("runtime_config", "dry_run")


def serve(config_path: Path) -> int:
    configure_logging()
    serve_config = load_json_object(config_path)
    reject_unknown_keys(serve_config, SERVE_KEYS, WHERE)
    dry_run = optional(serve_config, "dry_run", bool, False, WHERE)
    runtime = load_runtime_config(runtime_path_for_config(config_path))
    runtime.apply_environment()
    command_without_api_key = _judge_server_command(runtime)
    logger.info(shlex.join(command_without_api_key))
    if dry_run:
        return 0
    runtime.layout.create_directories()
    return subprocess.call(_with_api_key(command_without_api_key, runtime.judge_api_key()))


def _judge_server_command(runtime: RuntimeConfig) -> list[str]:
    judge_server = runtime.judge_server
    endpoint = require_local_http_endpoint(judge_server.base_url, where="judge_server.base_url")
    return vllm_serve_command(
        runtime.resolve_model(judge_server.model),
        endpoint,
        dtype=judge_server.dtype,
        max_model_len=judge_server.max_model_len,
        gpu_memory_utilization=judge_server.gpu_memory_utilization,
        tensor_parallel_size=judge_server.tensor_parallel_size,
        quantization=judge_server.quantization,
    )


def _with_api_key(command: list[str], api_key: str) -> list[str]:
    return [*command, "--api-key", api_key] if api_key else command
