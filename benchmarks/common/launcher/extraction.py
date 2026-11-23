from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError
from benchmarks.common.launcher.processes import run_job, server_alive, start_server, stop_server
from benchmarks.common.vllm_server import require_local_http_endpoint, vllm_serve_command
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.protocol import MEMORY_EXTRACTION_MAX_MODEL_LEN

RESERVED_VLLM_ARGS = frozenset(
    {
        "--host",
        "--port",
        "--model",
        "--max-model-len",
        "--gpu-memory-utilization",
        "--dtype",
        "--api-key",
        "--reasoning-parser",
    }
)
HEALTH_POLL_INTERVAL_S = 2


def extraction_command(
    run: dict[str, Any], runtime: RuntimeConfig
) -> tuple[list[str], MemoryRunConfig]:
    config = MemoryRunConfig.from_dict(run, runtime=runtime)
    endpoint = _local_endpoint(config)
    _reject_reserved_vllm_args(config.extraction.extra_vllm_args)
    command = vllm_serve_command(
        config.model,
        endpoint,
        dtype="auto",
        max_model_len=MEMORY_EXTRACTION_MAX_MODEL_LEN,
        gpu_memory_utilization=config.extraction.gpu_memory_utilization,
    )
    if config.extraction.structured_outputs:
        command += ["--structured-outputs-config", json.dumps(config.extraction.structured_outputs)]
    reasoning_parser = runtime.reasoning_parser(run["model"])
    if reasoning_parser:
        command += ["--reasoning-parser", reasoning_parser]
    command.extend(config.extraction.extra_vllm_args)
    return command, config


def _local_endpoint(config: MemoryRunConfig) -> SplitResult:
    endpoint = require_local_http_endpoint(config.memory_llm_base_url, where="mem0.llm_base_url")
    if endpoint.port != config.extraction.port or endpoint.path.rstrip("/") != "/v1":
        raise ConfigError("mem0.llm_base_url must match extraction.port and end in /v1.")
    return endpoint


def _reject_reserved_vllm_args(extra_vllm_args: Sequence[str]) -> None:
    option_names = {argument.split("=", 1)[0] for argument in extra_vllm_args}
    if option_names & RESERVED_VLLM_ARGS:
        raise ConfigError(
            "extraction.extra_vllm_args cannot override the configured model, endpoint, "
            "credentials or server settings."
        )


def validate_extraction_server(run: dict[str, Any], runtime: RuntimeConfig) -> None:
    extraction_command(run, runtime)


def run_extraction(
    run: dict[str, Any], runtime: RuntimeConfig, job: dict[str, Any], log_file: Path
) -> int:
    server_command, config = extraction_command(run, runtime)
    if _is_serving(config):
        raise ConfigError(f"An existing server is already using {config.memory_llm_base_url}.")
    server_log = log_file.with_suffix(".server.log")
    server_log.parent.mkdir(parents=True, exist_ok=True)
    with server_log.open("w", encoding="utf-8") as server_output:
        server = start_server(
            _with_api_key(server_command, config.memory_llm_api_key), server_output
        )
        try:
            _wait_until_serving(server, config, server_log)
            return run_job(job, log_file)
        finally:
            stop_server(server)


def _is_serving(config: MemoryRunConfig) -> bool:
    return server_alive(config.memory_llm_base_url, config.memory_llm_api_key)


def _with_api_key(command: list[str], api_key: str | None) -> list[str]:
    return [*command, "--api-key", api_key] if api_key else command


def _wait_until_serving(
    server: subprocess.Popen[Any], config: MemoryRunConfig, server_log: Path
) -> None:
    deadline = time.monotonic() + config.extraction.health_timeout_s
    while not _is_serving(config):
        if server.poll() is not None:
            raise RuntimeError(f"Extraction server exited; see {server_log}.")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Extraction server health timeout; see {server_log}.")
        time.sleep(HEALTH_POLL_INTERVAL_S)
