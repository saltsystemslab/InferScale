"""The temporary vLLM server that serves the answer model during Mem0 fact extraction."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError
from benchmarks.common.launcher.processes import run_job, server_alive, stop_server
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


def extraction_command(
    data: dict[str, Any], runtime: RuntimeConfig
) -> tuple[list[str], MemoryRunConfig]:
    """The server command for a run's extraction settings, and the parsed run config."""
    config = MemoryRunConfig.from_dict(data, runtime=runtime)
    endpoint = require_local_http_endpoint(config.memory_llm_base_url, where="mem0.llm_base_url")
    if endpoint.port != config.extraction.port or endpoint.path.rstrip("/") != "/v1":
        raise ConfigError("mem0.llm_base_url must match extraction.port and end in /v1.")
    extra_args = config.extraction.extra_vllm_args
    if any(value.split("=", 1)[0] in RESERVED_VLLM_ARGS for value in extra_args):
        raise ConfigError(
            "extraction.extra_vllm_args cannot override the configured model, endpoint, "
            "credentials or server settings."
        )
    command = vllm_serve_command(
        config.model,
        endpoint,
        dtype="auto",
        max_model_len=MEMORY_EXTRACTION_MAX_MODEL_LEN,
        gpu_memory_utilization=config.extraction.gpu_memory_utilization,
    )
    if config.extraction.structured_outputs:
        command += ["--structured-outputs-config", json.dumps(config.extraction.structured_outputs)]
    parser = runtime.reasoning_parser(data["model"])
    if parser:
        command += ["--reasoning-parser", parser]
    command.extend(extra_args)
    return command, config


def run_extraction(
    data: dict[str, Any], runtime: RuntimeConfig, job: dict[str, Any], log_file: Path
) -> int:
    """Start the extraction server, run the preembed job against it, then stop the server."""
    server_command, config = extraction_command(data, runtime)
    if server_alive(config.memory_llm_base_url, config.memory_llm_api_key):
        raise ConfigError(f"An existing server is already using {config.memory_llm_base_url}.")
    server_log = log_file.with_suffix(".server.log")
    server_log.parent.mkdir(parents=True, exist_ok=True)
    if config.memory_llm_api_key:
        server_command += ["--api-key", config.memory_llm_api_key]
    with server_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            server_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            _wait_until_serving(process, config, server_log)
            return run_job(job, log_file)
        finally:
            stop_server(process)


def _wait_until_serving(
    process: subprocess.Popen[Any], config: MemoryRunConfig, server_log: Path
) -> None:
    deadline = time.monotonic() + config.extraction.health_timeout_s
    while not server_alive(config.memory_llm_base_url, config.memory_llm_api_key):
        if process.poll() is not None:
            raise RuntimeError(f"Extraction server exited; see {server_log}.")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Extraction server health timeout; see {server_log}.")
        time.sleep(2)
